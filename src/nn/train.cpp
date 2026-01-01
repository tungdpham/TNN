/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/train.hpp"
#include "device/device_manager.hpp"
#include "threading/thread_wrapper.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace tnn {
void TrainingConfig::print_config() const {
  std::cout << "Training Configuration:" << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
  std::cout << "  LR Decay Factor: " << lr_decay_factor << std::endl;
  std::cout << "  LR Decay Interval (epochs): " << lr_decay_interval << std::endl;
  std::cout << "  Progress Print Interval (batches): " << progress_print_interval << std::endl;
  std::cout << "  Number of Threads: " << num_threads << std::endl;
  std::cout << "  Profiler Type: "
            << (profiler_type == ProfilerType::NONE
                    ? "None"
                    : (profiler_type == ProfilerType::NORMAL ? "Normal" : "Cumulative"))
            << std::endl;
  std::cout << "  Print Layer Profiling Info: " << (print_layer_profiling ? "Yes" : "No")
            << std::endl;
  std::cout << "  Print Layer Memory Usage: " << (print_layer_memory_usage ? "Yes" : "No")
            << std::endl;
  std::cout << "  Number of Microbatches: " << num_microbatches << std::endl;
  std::cout << "  Device Type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << std::endl;
}

void TrainingConfig::load_from_env() {
  // Get training parameters from environment or use defaults
  epochs = Env::get<int>("EPOCHS", DEFAULT_EPOCH);
  batch_size = Env::get<size_t>("BATCH_SIZE", DEFAULT_BATCH_SIZE);
  lr_decay_factor = Env::get<float>("LR_DECAY_FACTOR", DEFAULT_LR_DECAY_FACTOR);
  lr_decay_interval = Env::get<size_t>("LR_DECAY_INTERVAL", DEFAULT_LR_DECAY_INTERVAL);
  progress_print_interval = Env::get<int>("PROGRESS_PRINT_INTERVAL", DEFAULT_PRINT_INTERVAL);
  std::string profiler_type_str = Env::get<std::string>("PROFILER_TYPE", "NONE");
  if (profiler_type_str == "NORMAL") {
    profiler_type = ProfilerType::NORMAL;
  } else if (profiler_type_str == "CUMULATIVE") {
    profiler_type = ProfilerType::CUMULATIVE;
  } else {
    profiler_type = ProfilerType::NONE;
  }
  num_threads = Env::get<size_t>("NUM_THREADS", DEFAULT_NUM_THREADS);
  print_layer_profiling = Env::get<bool>("PRINT_LAYER_PROFILING", false);
  print_layer_memory_usage = Env::get<bool>("PRINT_LAYER_MEMORY_USAGE", false);
  num_microbatches = Env::get<size_t>("NUM_MICROBATCHES", 2);
  std::string device_type_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;
}

template <typename T>
ClassResult train_class_epoch(Sequential<T> &model, BaseDataLoader<T> &train_loader,
                              Optimizer<T> &optimizer, Loss<T> &loss_function,
                              const TrainingConfig &config) {
  Tensor<T> batch_data, batch_labels;
  std::cout << "Starting training epoch..." << std::endl;
  model.set_training(true);
  train_loader.shuffle();
  train_loader.reset();

  double total_loss = 0.0;
  double total_corrects = 0.0;
  int num_samples = 0;
  int num_batches = 0;
  const Device *model_device = model.get_device();

  Tensor<T> device_batch_data(model_device), device_batch_labels(model_device),
      loss_gradient(model_device);
  Tensor<T> predictions(model_device), backward_output(model_device);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    ++num_batches;
    num_samples += batch_data.shape()[0];
    device_batch_data = batch_data.to_device(model_device);
    model.forward(device_batch_data, predictions);
    device_batch_labels = batch_labels.to_device(model_device);

    T loss;
    loss_function.compute_loss(predictions, device_batch_labels, loss);
    int corrects = compute_class_corrects(predictions, device_batch_labels);

    total_loss += loss;
    total_corrects += corrects;

    loss_function.compute_gradient(predictions, device_batch_labels, loss_gradient);
    model.backward(loss_gradient, backward_output);

    optimizer.update();

    optimizer.clear_gradients();

    if (num_batches % config.progress_print_interval == 0) {
      if (model.is_profiling_enabled()) {
        if (config.print_layer_profiling)
          model.print_layers_profiling_info();
        if (config.print_layer_memory_usage)
          model.print_cache_memory_summary();
        model.print_profiling_summary();
      }
      std::cout << "Batch ID: " << num_batches << ", Batch's Loss: " << std::fixed
                << std::setprecision(4) << loss << ", Cumulative Accuracy: " << std::setprecision(2)
                << (total_corrects * 100.0 / num_samples) << "%" << std::endl;
    }
    if (model.is_profiling_enabled() && config.profiler_type == ProfilerType::NORMAL) {
      model.clear_profiling_data();
    }
  }
  std::cout << std::endl;

  const T avg_train_loss = static_cast<T>(total_loss / num_batches);
  const T avg_train_accuracy = static_cast<T>(total_corrects / num_samples);

  return {avg_train_loss, avg_train_accuracy};
}

template <typename T>
ClassResult validate_class_model(Sequential<T> &model, BaseDataLoader<T> &test_loader,
                                 Loss<T> &loss_function) {
  Tensor<T> batch_data, batch_labels;

  model.set_training(false);
  test_loader.reset();

  std::cout << "Starting validation..." << std::endl;
  double val_loss = 0.0;
  double val_corrects = 0.0;
  int val_batches = 0;
  const Device *model_device = model.get_device();

  Tensor<T> device_batch_data(model_device), device_batch_labels(model_device);
  Tensor<T> predictions(model_device);
  while (test_loader.get_next_batch(batch_data, batch_labels)) {
    model.forward(batch_data.to_device(model_device), predictions);

    device_batch_labels = batch_labels.to_device(model_device);
    T loss;
    loss_function.compute_loss(predictions, device_batch_labels, loss);
    val_loss += loss;
    val_corrects += compute_class_corrects(predictions, device_batch_labels);
    ++val_batches;
  }

  const T avg_val_loss = static_cast<T>(val_loss / val_batches);
  const T avg_val_accuracy = static_cast<T>(val_corrects / test_loader.size());

  return {avg_val_loss, avg_val_accuracy};
}

template <typename T>
void train_classification_model(Sequential<T> &model, BaseDataLoader<T> &train_loader,
                                BaseDataLoader<T> &test_loader,
                                std::unique_ptr<Optimizer<T>> optimizer,
                                std::unique_ptr<Loss<T>> loss_function,
                                std::unique_ptr<Scheduler<T>> scheduler,
                                const TrainingConfig &config) {
  optimizer->attach(model.parameters(), model.gradients());
  Tensor<T> batch_data, batch_labels;

  train_loader.prepare_batches(config.batch_size);
  test_loader.prepare_batches(config.batch_size);

  if (config.profiler_type == ProfilerType::NONE) {
    model.enable_profiling(false);
  } else if (config.profiler_type == ProfilerType::NORMAL ||
             config.profiler_type == ProfilerType::CUMULATIVE) {
    model.enable_profiling(true);
  }

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;

  std::vector<size_t> data_shape = train_loader.get_data_shape();

  model.print_summary({config.batch_size, data_shape[0], data_shape[1], data_shape[2]});

  T best_val_accuracy = 0.0;

  ThreadWrapper thread_wrapper({config.num_threads});

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << std::endl;

      // train phrase
      auto train_start = std::chrono::high_resolution_clock::now();
      auto [avg_train_loss, avg_train_accuracy] =
          train_class_epoch(model, train_loader, *optimizer, *loss_function, config);
      auto train_end = std::chrono::high_resolution_clock::now();
      auto train_epoch_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] =
          validate_class_model(model, test_loader, *loss_function);

      if (avg_val_accuracy > best_val_accuracy) {
        best_val_accuracy = avg_val_accuracy;
        std::cout << "New best validation accuracy: " << std::fixed << std::setprecision(2)
                  << best_val_accuracy * 100.0 << "%" << std::endl;
        try {
          model.save_to_file("model_snapshots/" + model.name());
          std::cout << "Model saved to " << "model_snapshots/" + model.name() << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "Error saving model: " << e.what() << std::endl;
        }
      }

      std::cout << std::string(60, '-') << std::endl;
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " completed in "
                << train_epoch_duration.count() << "ms" << std::endl;
      std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                << ", Accuracy: " << std::setprecision(2) << avg_train_accuracy * 100.0 << "%"
                << std::endl;
      std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4) << avg_val_loss
                << ", Accuracy: " << std::setprecision(2) << avg_val_accuracy * 100.0 << "%"
                << std::endl;
      std::cout << std::string(60, '=') << std::endl;

      if (model.is_profiling_enabled()) {
        model.clear_profiling_data();
      }

      // Step the scheduler if provided
      if (scheduler) {
        scheduler->step();
      }

      if ((epoch + 1) % 5 == 0) {
        thread_wrapper.clean_buffers();
      }

      // re prepare batches to reapply augmentation
      train_loader.prepare_batches(config.batch_size);
      test_loader.prepare_batches(config.batch_size);

      std::cout << get_memory_usage_kb() / 1024 << " MB of memory used." << std::endl;
    }
  });
}

template <typename T>
RegResult train_reg_epoch(Sequential<T> &model, RegressionDataLoader<T> &train_loader,
                          Optimizer<T> &optimizer, Loss<T> &loss_function,
                          const TrainingConfig &config) {
  Tensor<T> batch_data, batch_labels;
  std::cout << "Starting training epoch..." << std::endl;
  model.set_training(true);
  train_loader.shuffle();
  train_loader.reset();

  double total_loss = 0.0;
  int num_batches = 0;

  const Device *model_device = model.get_device();

  Tensor<T> device_batch_data(model_device), device_batch_labels(model_device),
      loss_gradient(model_device);
  Tensor<T> predictions, backward_output;

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    ++num_batches;
    device_batch_data = batch_data.to_device(model_device);
    device_batch_labels = batch_labels.to_device(model_device);

    model.forward(device_batch_data, predictions);

    T loss;
    loss_function.compute_loss(predictions, device_batch_labels, loss);
    total_loss += loss;

    loss_function.compute_gradient(predictions, device_batch_labels, loss_gradient);
    model.backward(loss_gradient, backward_output);

    optimizer.update();

    optimizer.clear_gradients();

    if (num_batches % config.progress_print_interval == 0) {
      if (model.is_profiling_enabled()) {
        model.print_layers_profiling_info();
        model.print_profiling_summary();
      }
      std::cout << "Batch ID: " << num_batches << ", Batch's Loss: " << std::fixed
                << std::setprecision(4) << loss << ", Batch's Error: " << std::setprecision(2)
                << std::endl;
    }
    if (model.is_profiling_enabled()) {
      model.clear_profiling_data();
    }
  }
  std::cout << std::endl;

  const T avg_train_loss = static_cast<T>(total_loss / num_batches);

  return {avg_train_loss, 0.0};
}

template <typename T>
RegResult validate_reg_model(Sequential<T> &model, RegressionDataLoader<T> &test_loader,
                             Loss<T> &loss_function) {
  Tensor<T> batch_data, batch_labels;

  model.set_training(false);
  test_loader.reset();

  std::cout << "Starting validation..." << std::endl;
  double val_loss = 0.0;
  int val_batches = 0;

  const Device *model_device = model.get_device();

  Tensor<T> device_batch_data(model_device), device_batch_labels(model_device);
  Tensor<T> predictions;

  while (test_loader.get_next_batch(batch_data, batch_labels)) {
    device_batch_data = batch_data.to_device(model_device);
    device_batch_labels = batch_labels.to_device(model_device);

    model.forward(device_batch_data, predictions);

    T loss;
    loss_function.compute_loss(predictions, device_batch_labels, loss);
    val_loss += loss;
    ++val_batches;
  }

  const T avg_val_loss = static_cast<T>(val_loss / val_batches);

  return {avg_val_loss, 0.0};
}

template <typename T>
void train_regression_model(Sequential<T> &model, RegressionDataLoader<T> &train_loader,
                            RegressionDataLoader<T> &test_loader,
                            std::unique_ptr<Optimizer<T>> optimizer,
                            std::unique_ptr<Loss<T>> loss_function,
                            std::unique_ptr<Scheduler<T>> scheduler, const TrainingConfig &config) {
  Tensor<T> batch_data, batch_labels;

  train_loader.prepare_batches(config.batch_size);
  test_loader.prepare_batches(config.batch_size);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;

  // Attach optimizer to model parameters
  optimizer->attach(model.parameters(), model.gradients());

  auto [best_val_loss, best_val_error] = validate_reg_model(model, test_loader, *loss_function);

  ThreadWrapper thread_wrapper({config.num_threads});
  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << std::endl;

      // train phrase
      auto train_start = std::chrono::high_resolution_clock::now();
      auto [avg_train_loss, avg_train_accuracy] =
          train_reg_epoch(model, train_loader, *optimizer, *loss_function, config);
      auto train_end = std::chrono::high_resolution_clock::now();
      auto epoch_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

      // validation phrase
      auto val_start = std::chrono::high_resolution_clock::now();
      auto [avg_val_loss, avg_val_accuracy] =
          validate_reg_model(model, test_loader, *loss_function);
      auto val_end = std::chrono::high_resolution_clock::now();
      epoch_duration += std::chrono::duration_cast<std::chrono::milliseconds>(val_end - val_start);

      if (avg_val_loss < best_val_loss) {
        best_val_loss = avg_val_loss;
        std::cout << "New best validation loss: " << std::fixed << std::setprecision(4)
                  << best_val_loss << std::endl;
        try {
          model.save_to_file("model_snapshots/" + model.name());
          std::cout << "Model saved to " << "model_snapshots/" + model.name() << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "Error saving model: " << e.what() << std::endl;
        }
      }

      std::cout << std::string(60, '-') << std::endl;
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " completed in "
                << epoch_duration.count() << "ms" << std::endl;
      std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                << ", Accuracy: " << std::setprecision(2) << avg_train_accuracy * 100.0 << "%"
                << std::endl;
      std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4) << avg_val_loss
                << ", Accuracy: " << std::setprecision(2) << avg_val_accuracy * 100.0 << "%"
                << std::endl;
      std::cout << std::string(60, '=') << std::endl;

      if (model.is_profiling_enabled()) {
        model.clear_profiling_data();
      }

      // Step the scheduler if provided
      if (scheduler) {
        scheduler->step();
      }

      if ((epoch + 1) % 5 == 0) {
        thread_wrapper.clean_buffers();
      }

      // re prepare batches to reapply augmentation
      train_loader.prepare_batches(config.batch_size);

      std::cout << get_memory_usage_kb() / 1024 << " MB of memory used." << std::endl;
    }
  });
}

template ClassResult train_class_epoch<float>(Sequential<float> &model,
                                              BaseDataLoader<float> &train_loader,
                                              Optimizer<float> &optimizer,
                                              Loss<float> &loss_function,
                                              const TrainingConfig &config);

template ClassResult validate_class_model<float>(Sequential<float> &model,
                                                 BaseDataLoader<float> &test_loader,
                                                 Loss<float> &loss_function);

template void train_classification_model<float>(Sequential<float> &model,
                                                BaseDataLoader<float> &train_loader,
                                                BaseDataLoader<float> &test_loader,
                                                std::unique_ptr<Optimizer<float>> optimizer,
                                                std::unique_ptr<Loss<float>> loss_function,
                                                std::unique_ptr<Scheduler<float>> scheduler,
                                                const TrainingConfig &config);

template RegResult train_reg_epoch<float>(Sequential<float> &model,
                                          RegressionDataLoader<float> &train_loader,
                                          Optimizer<float> &optimizer, Loss<float> &loss_function,
                                          const TrainingConfig &config);

template RegResult validate_reg_model<float>(Sequential<float> &model,
                                             RegressionDataLoader<float> &test_loader,
                                             Loss<float> &loss_function);

template void train_regression_model<float>(Sequential<float> &model,
                                            RegressionDataLoader<float> &train_loader,
                                            RegressionDataLoader<float> &test_loader,
                                            std::unique_ptr<Optimizer<float>> optimizer,
                                            std::unique_ptr<Loss<float>> loss_function,
                                            std::unique_ptr<Scheduler<float>> scheduler,
                                            const TrainingConfig &config);

} // namespace tnn
