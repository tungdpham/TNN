/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "data_loading/data_loader.hpp"
#include "nn/sequential.hpp"
#include "utils/memory.hpp"
#include "utils/utils_extended.hpp"
#ifdef USE_TBB
#include <tbb/info.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_arena.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_TBB
void tbb_cleanup() {
  // Clean all buffers
  scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS, 0);
}
#endif

constexpr uint32_t DEFAULT_NUM_THREADS = 8; // Typical number of P-Cores on laptop CPUs

struct TrainingConfig {
  int epochs = 10;
  size_t batch_size = 32;
  float lr_decay_factor = 0.9f;
  size_t lr_decay_interval = 5; // in epochs
  int progress_print_interval = 100;
  uint32_t num_threads = 8; // Typical number of P-Cores on laptop CPUs
};

struct ClassResult {
  float avg_loss = 0.0f;
  float avg_accuracy = 0.0f;
};

ClassResult train_class_epoch(tnn::Sequential<float> &model,
                              data_loading::ImageClassificationDataLoader<float> &train_loader,
                              const TrainingConfig &config = TrainingConfig()) {
  Tensor<float> batch_data, batch_labels, predictions;
  std::cout << "Starting training epoch..." << std::endl;
  model.set_training(true);
  train_loader.shuffle();
  train_loader.reset();

  double total_loss = 0.0;
  double total_accuracy = 0.0;
  int num_batches = 0;

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    ++num_batches;

    predictions = model.forward(batch_data);
    predictions.apply_softmax();

    const float loss = model.loss_function()->compute_loss(predictions, batch_labels);
    const float accuracy = utils::compute_class_accuracy<float>(predictions, batch_labels);

    total_loss += loss;
    total_accuracy += accuracy;

    const Tensor<float> loss_gradient =
        model.loss_function()->compute_gradient(predictions, batch_labels);
    model.backward(loss_gradient);

    model.update_parameters();

    if (num_batches % config.progress_print_interval == 0) {
      if (model.is_profiling_enabled()) {
        model.print_profiling_summary();
      }
      std::cout << "Batch ID: " << num_batches << ", Batch's Loss: " << std::fixed
                << std::setprecision(4) << loss << ", Batch's Accuracy: " << std::setprecision(2)
                << accuracy * 100.0f << "%" << std::endl;
    }
    if (model.is_profiling_enabled()) {
      model.clear_profiling_data();
    }
  }
  std::cout << std::endl;

  const float avg_train_loss = static_cast<float>(total_loss / num_batches);
  const float avg_train_accuracy = static_cast<float>(total_accuracy / num_batches);

  return {avg_train_loss, avg_train_accuracy};
}

ClassResult validate_class_model(tnn::Sequential<float> &model,
                                 data_loading::ImageClassificationDataLoader<float> &test_loader) {
  Tensor<float> batch_data, batch_labels, predictions;

  model.set_training(false);
  test_loader.reset();

  std::cout << "Starting validation..." << std::endl;
  double val_loss = 0.0;
  double val_accuracy = 0.0;
  int val_batches = 0;

  while (test_loader.get_next_batch(batch_data, batch_labels)) {
    predictions = model.forward(batch_data);
    predictions.apply_softmax();

    val_loss += model.loss_function()->compute_loss(predictions, batch_labels);
    val_accuracy += utils::compute_class_accuracy<float>(predictions, batch_labels);
    ++val_batches;
  }

  const float avg_val_loss = static_cast<float>(val_loss / val_batches);
  const float avg_val_accuracy = static_cast<float>(val_accuracy / val_batches);

  return {avg_val_loss, avg_val_accuracy};
}

void train_classification_model(tnn::Sequential<float> &model,
                                data_loading::ImageClassificationDataLoader<float> &train_loader,
                                data_loading::ImageClassificationDataLoader<float> &test_loader,
                                const TrainingConfig &config = TrainingConfig()) {

  Tensor<float> batch_data, batch_labels, predictions;

  train_loader.prepare_batches(config.batch_size);
  test_loader.prepare_batches(config.batch_size);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;

  std::vector<size_t> image_shape = train_loader.get_image_shape();

#ifdef USE_MKL
  std::cout << "Setting MKL number of threads to: " << 8 << std::endl;
  mkl_set_threading_layer(MKL_THREADING_TBB);
#endif

  model.print_summary({config.batch_size, image_shape[0], image_shape[1], image_shape[2]});

  auto [best_val_loss, best_val_accuracy] = validate_class_model(model, test_loader);

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(config.num_threads));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&] {
#endif
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << std::endl;

      // train phrase
      auto train_start = std::chrono::high_resolution_clock::now();
      auto [avg_train_loss, avg_train_accuracy] = train_class_epoch(model, train_loader, config);
      auto train_end = std::chrono::high_resolution_clock::now();
      auto epoch_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

      // validation phrase
      auto val_start = std::chrono::high_resolution_clock::now();
      auto [avg_val_loss, avg_val_accuracy] = validate_class_model(model, test_loader);
      auto val_end = std::chrono::high_resolution_clock::now();
      epoch_duration += std::chrono::duration_cast<std::chrono::milliseconds>(val_end - val_start);

      if (avg_val_accuracy > best_val_accuracy) {
        best_val_accuracy = avg_val_accuracy;
        std::cout << "New best validation accuracy: " << std::fixed << std::setprecision(2)
                  << best_val_accuracy * 100.0f << "%" << std::endl;
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
                << ", Accuracy: " << std::setprecision(2) << avg_train_accuracy * 100.0f << "%"
                << std::endl;
      std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4) << avg_val_loss
                << ", Accuracy: " << std::setprecision(2) << avg_val_accuracy * 100.0f << "%"
                << std::endl;
      std::cout << std::string(60, '=') << std::endl;

      if (model.is_profiling_enabled()) {
        model.clear_profiling_data();
      }

      // learning rate decay
      if ((epoch + 1) % config.lr_decay_interval == 0) {
        const float current_lr = model.optimizer()->get_learning_rate();
        const float new_lr = current_lr * config.lr_decay_factor;
        model.optimizer()->set_learning_rate(new_lr);
        std::cout << "Learning rate decayed: " << std::fixed << std::setprecision(6) << current_lr
                  << " -> " << new_lr << std::endl;
      }

      if ((epoch + 1) % 5 == 0) {
        tbb_cleanup();
      }

      // re prepare batches to reapply augmentation
      train_loader.prepare_batches(config.batch_size);

      std::cout << utils::get_memory_usage_kb() / 1024 << " MB of memory used." << std::endl;
    }

#ifdef USE_TBB
  });
#endif
}

struct RegResult {
  float avg_loss = 0.0f;
  float avg_error = 0.0f;
};

RegResult train_reg_epoch(tnn::Sequential<float> &model,
                          data_loading::RegressionDataLoader<float> &train_loader,
                          const TrainingConfig &config = TrainingConfig()) {
  Tensor<float> batch_data, batch_labels, predictions;
  std::cout << "Starting training epoch..." << std::endl;
  model.set_training(true);
  train_loader.shuffle();
  train_loader.reset();

  double total_loss = 0.0;
  int num_batches = 0;

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    ++num_batches;

    predictions = model.forward(batch_data);

    const float loss = model.loss_function()->compute_loss(predictions, batch_labels);

    total_loss += loss;

    const Tensor<float> loss_gradient =
        model.loss_function()->compute_gradient(predictions, batch_labels);
    model.backward(loss_gradient);

    model.update_parameters();

    if (num_batches % config.progress_print_interval == 0) {
      if (model.is_profiling_enabled()) {
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

  const float avg_train_loss = static_cast<float>(total_loss / num_batches);

  return {avg_train_loss, 0.0f};
}

RegResult validate_reg_model(tnn::Sequential<float> &model,
                             data_loading::RegressionDataLoader<float> &test_loader) {
  Tensor<float> batch_data, batch_labels, predictions;

  model.set_training(false);
  test_loader.reset();

  std::cout << "Starting validation..." << std::endl;
  double val_loss = 0.0;
  int val_batches = 0;

  while (test_loader.get_next_batch(batch_data, batch_labels)) {
    predictions = model.forward(batch_data);

    val_loss += model.loss_function()->compute_loss(predictions, batch_labels);
    ++val_batches;
  }

  const float avg_val_loss = static_cast<float>(val_loss / val_batches);

  return {avg_val_loss, 0.0f};
}

void train_regression_model(tnn::Sequential<float> &model,
                            data_loading::RegressionDataLoader<float> &train_loader,
                            data_loading::RegressionDataLoader<float> &test_loader,
                            const TrainingConfig &config = TrainingConfig()) {

  Tensor<float> batch_data, batch_labels, predictions;

  train_loader.prepare_batches(config.batch_size);
  test_loader.prepare_batches(config.batch_size);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;

  auto [best_val_loss, best_val_error] = validate_reg_model(model, test_loader);

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(config.num_threads));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&] {
#endif
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << std::endl;

      // train phrase
      auto train_start = std::chrono::high_resolution_clock::now();
      auto [avg_train_loss, avg_train_accuracy] = train_reg_epoch(model, train_loader, config);
      auto train_end = std::chrono::high_resolution_clock::now();
      auto epoch_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

      // validation phrase
      auto val_start = std::chrono::high_resolution_clock::now();
      auto [avg_val_loss, avg_val_accuracy] = validate_reg_model(model, test_loader);
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
                << ", Accuracy: " << std::setprecision(2) << avg_train_accuracy * 100.0f << "%"
                << std::endl;
      std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4) << avg_val_loss
                << ", Accuracy: " << std::setprecision(2) << avg_val_accuracy * 100.0f << "%"
                << std::endl;
      std::cout << std::string(60, '=') << std::endl;

      if (model.is_profiling_enabled()) {
        model.clear_profiling_data();
      }

      // learning rate decay
      if ((epoch + 1) % config.lr_decay_interval == 0) {
        const float current_lr = model.optimizer()->get_learning_rate();
        const float new_lr = current_lr * config.lr_decay_factor;
        model.optimizer()->set_learning_rate(new_lr);
        std::cout << "Learning rate decayed: " << std::fixed << std::setprecision(6) << current_lr
                  << " -> " << new_lr << std::endl;
      }

      if ((epoch + 1) % 5 == 0) {
        tbb_cleanup();
      }

      // re prepare batches to reapply augmentation
      train_loader.prepare_batches(config.batch_size);

      std::cout << utils::get_memory_usage_kb() / 1024 << " MB of memory used." << std::endl;
    }

#ifdef USE_TBB
  });
#endif
}