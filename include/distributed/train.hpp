/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <unordered_map>

#include "coordinator.hpp"
#include "nn/csv_logger.hpp"
#include "nn/train.hpp"
#include "tensor/tensor_ops.hpp"
#include "threading/thread_wrapper.hpp"
#include "type/type.hpp"

namespace tnn {

inline Result train_semi_async_epoch(Coordinator &coordinator,
                                     std::unique_ptr<BaseDataLoader> &train_loader,
                                     const std::unique_ptr<Loss> &criterion,
                                     const TrainingConfig &config, CsvLogger &logger, int epoch) {
  train_loader->shuffle();
  train_loader->reset();

  Tensor batch_data, batch_labels;
  size_t batch_index = 0;
  int accumulation_steps = 0;
  auto epoch_start = std::chrono::high_resolution_clock::now();
  float total_loss = 0.0f;
  int total_corrects = 0;
  size_t total_samples = 0;
  coordinator.set_training(true);
  while (train_loader->get_batch(config.batch_size, batch_data, batch_labels)) {
    // Split batch into micro-batches
    Vec<Tensor> micro_batch_inputs;
    ops::split(batch_data, micro_batch_inputs, config.num_microbatches);
    Vec<Tensor> micro_batch_labels;
    ops::split(batch_labels, micro_batch_labels, config.num_microbatches);

    auto process_start = std::chrono::high_resolution_clock::now();
    // Perform forward, compute loss, and backward asynchronously.
    auto [loss, corrects] =
        coordinator.async_train_batch(micro_batch_inputs, micro_batch_labels, criterion);
    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

    total_loss += loss;
    total_corrects += corrects;
    total_samples += batch_labels->dimension(0);
    accumulation_steps++;
    if (accumulation_steps == config.gradient_accumulation_steps) {
      coordinator.update_parameters();
      accumulation_steps = 0;
    }

    // Log batch metrics to CSV.
    {
      long time_ms = process_duration.count() / 1000;  // us -> ms
      double acc_pct =
          total_samples > 0 ? static_cast<double>(total_corrects) / total_samples * 100.0 : 0.0;

      std::unordered_map<std::string, double> metrics;
      if (config.log_mode.log_loss) {
        metrics["loss"] = loss;
      }
      if (config.log_mode.log_accuracy) {
        metrics["accuracy_pct"] = acc_pct;
      }
      metrics["time_ms"] = time_ms;

      logger.log_batch(epoch, static_cast<int>(batch_index + 1), metrics);
    }

    if ((batch_index + 1) % config.progress_print_interval == 0) {
      std::cout << "Batch " << (batch_index + 1) << " Loss: " << loss << ", Cummulative Accuracy: "
                << (static_cast<double>(total_corrects) / total_samples * 100.0f) << "%"
                << ", Processing Time: " << process_duration.count() << " us" << std::endl;
      if (config.profiler_type != ProfilerType::NONE) {
        coordinator.print_profiling();
      }
    }
    if (config.profiler_type != ProfilerType::NONE) {
      coordinator.clear_profiling();
    }
    ++batch_index;
  }

  auto epoch_end = std::chrono::high_resolution_clock::now();
  auto epoch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
  std::cout << "\nEpoch completed in " << epoch_duration.count() << " milliseconds" << std::endl;
  return {total_loss / batch_index, -1.0f};
}

inline Result validate_semi_async_epoch(Coordinator &coordinator,
                                        std::unique_ptr<BaseDataLoader> &val_loader,
                                        const std::unique_ptr<Loss> &criterion,
                                        const TrainingConfig &config) {
  val_loader->reset();

  Tensor batch_data, batch_labels;
  float total_val_loss = 0.0f;
  float total_val_correct = 0.0f;
  int val_batches = 0;

  coordinator.set_training(false);

  while (val_loader->get_batch(config.batch_size, batch_data, batch_labels)) {
    Vec<Tensor> micro_batch_inputs{std::move(batch_data)};
    Vec<Tensor> micro_batch_labels{std::move(batch_labels)};
    auto [val_loss, val_correct] =
        coordinator.async_val_batch(micro_batch_inputs, micro_batch_labels, criterion);
    total_val_loss += val_loss;
    total_val_correct += val_correct;
    ++val_batches;
  }

  std::cout << "Validation completed." << std::endl;
  std::cout << "Average Validation Loss: " << (total_val_loss / val_batches)
            << ", Average Validation Accuracy: "
            << (total_val_correct / val_loader->size()) * 100.0f << "%" << std::endl;
  return {total_val_loss / val_batches, (total_val_correct / val_loader->size()) * 100.0f};
}

inline void train_model(Coordinator &coordinator, std::unique_ptr<BaseDataLoader> &train_loader,
                        std::unique_ptr<BaseDataLoader> &val_loader,
                        const std::unique_ptr<Loss> &criterion,
                        TrainingConfig config = TrainingConfig()) {
  coordinator.start_profiling();
  ThreadWrapper thread_wrapper({config.num_threads});
  CsvLogger logger(config.model_name, config.log_dir, &config.log_mode);

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs << " ===" << std::endl;
      auto [train_loss, train_acc] =
          train_semi_async_epoch(coordinator, train_loader, criterion, config, logger, epoch + 1);

      auto [val_loss, val_acc] =
          validate_semi_async_epoch(coordinator, val_loader, criterion, config);

      std::unordered_map<std::string, double> metrics;
      if (config.log_mode.log_loss) {
        metrics["train_loss"] = train_loss;
        metrics["val_loss"] = val_loss;
      }
      if (config.log_mode.log_accuracy) {
        metrics["train_accuracy_pct"] = train_acc;
        metrics["val_accuracy_pct"] = val_acc;
      }
      logger.log_epoch(epoch + 1, metrics);
    }

    coordinator.fetch_profiling();
    coordinator.print_logs();
    logger.flush();
  });
}

}  // namespace tnn