/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/train.hpp"
#include "nn/accuracy.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
#include "utils/env.hpp"
#include "utils/memory.hpp"
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

namespace tnn {
void TrainingConfig::print_config() const {
  std::cout << "Training Configuration:" << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
  std::cout << "  Max Steps: " << max_steps << std::endl;
  std::cout << "  Initial Learning Rate: " << lr_initial << std::endl;
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
  max_steps = Env::get<uint64_t>("MAX_STEPS", -1); // -1 for no limit
  lr_initial = Env::get<float>("LR_INITIAL", 0.001f);
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
  num_threads = Env::get<int64_t>("NUM_THREADS", DEFAULT_NUM_THREADS);
  print_layer_profiling = Env::get<bool>("PRINT_LAYER_PROFILING", false);
  print_layer_memory_usage = Env::get<bool>("PRINT_LAYER_MEMORY_USAGE", false);
  num_microbatches = Env::get<size_t>("NUM_MICROBATCHES", 2);
  std::string device_type_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;
}

Result train_epoch(Sequential &model, BaseDataLoader &train_loader, Optimizer &optimizer,
                   Loss &criterion, const TrainingConfig &config) {
  auto train_start = std::chrono::high_resolution_clock::now();
  Tensor batch_data = make_tensor_from_dtype(config.dtype),
         batch_labels = make_tensor_from_dtype(config.dtype);
  std::cout << "Starting training epoch..." << std::endl;
  model.set_training(true);
  // train_loader.shuffle();
  train_loader.reset();

  float total_loss = 0.0;
  double total_corrects = 0.0;
  size_t cur_samples = 0;
  int num_batches = 0;
  const Device *model_device = model.get_device();

  Tensor device_labels = make_tensor_from_dtype(config.dtype, {1}, model_device);
  Tensor loss_gradient = make_tensor_from_dtype(config.dtype, {1}, model_device);
  Tensor predictions = make_tensor_from_dtype(config.dtype, {1}, model_device),
         backward_output = make_tensor_from_dtype(config.dtype, {1}, model_device);

  std::cout << "Training batches: " << train_loader.size() << std::endl;
  while (train_loader.get_batch(config.batch_size, batch_data, batch_labels) &&
         (config.max_steps == -1 || num_batches < config.max_steps)) {
    ++num_batches;
    cur_samples += batch_data->dimension(0);
    device_labels = batch_labels->to_device(model_device);
    model.forward(batch_data, predictions);

    float loss;
    criterion.compute_loss(predictions, device_labels, loss);

    total_loss += loss;
    total_corrects += compute_class_corrects(predictions, device_labels);

    criterion.compute_gradient(predictions, device_labels, loss_gradient);

    model.backward(loss_gradient, backward_output);

    optimizer.update();

    optimizer.clear_gradients();

    if (num_batches % config.progress_print_interval == 0) {
      if (model.is_profiling_enabled()) {
        model.print_profiling_info();
      }
      std::cout << "Batch ID: " << num_batches << ", Batch's Loss: " << std::fixed
                << std::setprecision(4) << loss << ", Cumulative Accuracy: " << std::setprecision(2)
                << (total_corrects * 100.0 / cur_samples) << "%" << std::endl;
    }
    if (model.is_profiling_enabled() && config.profiler_type == ProfilerType::NORMAL) {
      model.reset_profiling_info();
    }
  }
  std::cout << std::endl;

  const double avg_train_loss = total_loss / num_batches;
  const double avg_train_accuracy = total_corrects / cur_samples;

  auto train_end = std::chrono::high_resolution_clock::now();
  auto train_epoch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
  std::cout << "Training epoch completed in " << train_epoch_duration.count() << "ms" << std::endl;
  return {avg_train_loss, avg_train_accuracy};
}

Result validate_model(Sequential &model, BaseDataLoader &val_loader, Loss &criterion,
                      const TrainingConfig &config) {
  Tensor batch_data = make_tensor_from_dtype(config.dtype),
         batch_labels = make_tensor_from_dtype(config.dtype);

  model.set_training(false);
  val_loader.reset();

  std::cout << "Starting validation..." << std::endl;
  double val_loss = 0.0;
  double val_corrects = 0.0;
  int val_batches = 0;
  const Device *model_device = model.get_device();

  Tensor device_batch_labels = make_tensor_from_dtype(config.dtype, {}, model_device);
  Tensor predictions = make_tensor_from_dtype(config.dtype, {}, model_device);

  while (val_loader.get_batch(config.batch_size, batch_data, batch_labels)) {
    model.forward(batch_data, predictions);

    device_batch_labels = batch_labels->to_device(model_device);
    float loss;
    criterion.compute_loss(predictions, device_batch_labels, loss);
    val_loss += loss;
    val_corrects += compute_class_corrects(predictions, device_batch_labels);
    ++val_batches;
  }

  const double avg_val_loss = val_loss / val_batches;
  const double avg_val_accuracy = val_corrects / val_loader.size();

  return {avg_val_loss, avg_val_accuracy};
}

void train_val(Sequential &model, BaseDataLoader &train_loader, BaseDataLoader &val_loader,
               std::unique_ptr<Optimizer> &optimizer, const std::unique_ptr<Loss> &criterion,
               const std::unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  ThreadWrapper thread_wrapper({config.num_threads});

  double best_val_accuracy = 0.0;

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << std::endl;

      // train phrase
      auto [avg_train_loss, avg_train_accuracy] =
          train_epoch(model, train_loader, *optimizer, *criterion, config);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] = validate_model(model, val_loader, *criterion, config);

      if (avg_val_accuracy > best_val_accuracy) {
        best_val_accuracy = avg_val_accuracy;
        std::cout << "New best validation accuracy: " << std::fixed << std::setprecision(2)
                  << best_val_accuracy * 100.0 << "%" << std::endl;
        try {
          std::string filepath = "model_snapshots/" + model.name();
          std::ofstream file(filepath, std::ios::binary);
          if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
          }
          model.save_state(file);
          file.close();
          std::cout << "Model saved to " << filepath << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "Error saving model: " << e.what() << std::endl;
        }
      }

      std::cout << std::string(60, '-') << std::endl;
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << std::endl;
      std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                << ", Accuracy: " << std::setprecision(2) << avg_train_accuracy * 100.0 << "%"
                << std::endl;
      std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4) << avg_val_loss
                << ", Accuracy: " << std::setprecision(2) << avg_val_accuracy * 100.0 << "%"
                << std::endl;
      std::cout << std::string(60, '=') << std::endl;

      if (model.is_profiling_enabled()) {
        model.reset_profiling_info();
      }

      if (scheduler) {
        scheduler->step();
      }

      if ((epoch + 1) % 5 == 0) {
        thread_wrapper.clean_buffers();
      }

      std::cout << get_memory_usage_kb() / 1024 << " MB of memory used." << std::endl;
    }
  });
}

void train_step(Sequential &model, BaseDataLoader &train_loader,
                const std::unique_ptr<Optimizer> &optimizer, const std::unique_ptr<Loss> &criterion,
                const std::unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  ThreadWrapper thread_wrapper({config.num_threads});

  Tensor batch_data, batch_labels;
  std::cout << "Starting training epoch..." << std::endl;
  model.set_training(true);
  train_loader.shuffle();
  train_loader.reset();

  const Device *model_device = model.get_device();

  Tensor loss_gradient = make_tensor<float>({1}, model_device);
  Tensor predictions = make_tensor<float>({1}, model_device);
  Tensor backward_output = make_tensor<float>({1}, model_device);

  thread_wrapper.execute([&]() -> void {
    for (int steps = 0; steps < config.max_steps; ++steps) {
      if (!train_loader.get_batch(config.batch_size, batch_data, batch_labels)) {
        break;
      }
      model.forward(batch_data, predictions);

      float loss;
      criterion->compute_loss(predictions, batch_labels, loss);

      criterion->compute_gradient(predictions, batch_labels, loss_gradient);

      model.backward(loss_gradient, backward_output);

      optimizer->update();
      optimizer->clear_gradients();

      if (steps % config.progress_print_interval == 0) {
        if (model.is_profiling_enabled()) {
          model.print_profiling_info();
        }
        std::cout << "Batch ID: " << steps << ", Batch's Loss: " << std::fixed
                  << std::setprecision(4) << loss << std::endl;
      }
      if (model.is_profiling_enabled() && config.profiler_type == ProfilerType::NORMAL) {
        model.reset_profiling_info();
      }
    }

    // save model
    try {
      std::string filepath = "model_snapshots/" + model.name();
      std::ofstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
      }
      model.save_state(file);
      file.close();
      std::cout << "Model saved to " << filepath << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error saving model: " << e.what() << std::endl;
    }
  });
}

void train_model(Sequential &model, BaseDataLoader &train_loader, BaseDataLoader &val_loader,
                 std::unique_ptr<Optimizer> &optimizer, std::unique_ptr<Loss> &criterion,
                 std::unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  optimizer->attach(model.parameters(), model.gradients());
  Tensor batch_data, batch_labels;

  if (config.profiler_type == ProfilerType::NONE) {
    model.enable_profiling(false);
  } else if (config.profiler_type == ProfilerType::NORMAL ||
             config.profiler_type == ProfilerType::CUMULATIVE) {
    model.enable_profiling(true);
  }

  std::cout << "Training batches: " << train_loader.size() << std::endl;
  std::cout << "Validation batches: " << val_loader.size() << std::endl;

  std::vector<size_t> data_shape = train_loader.get_data_shape();
  data_shape.insert(data_shape.begin(), config.batch_size); // add batch dimension
  model.print_summary(data_shape);

  bool is_val = config.max_steps == -1;

  if (is_val) {
    train_val(model, train_loader, val_loader, optimizer, criterion, scheduler, config);
  } else {
    train_step(model, train_loader, optimizer, criterion, scheduler, config);
  }
}

} // namespace tnn
