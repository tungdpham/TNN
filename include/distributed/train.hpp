/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cmath>
#include <iomanip>
#include <memory>
#include <unordered_map>

#include "coordinator.hpp"
#include "data_loading/batch_prefetcher.hpp"
#include "nn/csv_logger.hpp"
#include "nn/train.hpp"
#include "tensor/tensor_ops.hpp"
#include "threading/thread_wrapper.hpp"
#include "type/type.hpp"

namespace tnn {

inline bool get_next_train_batch(BaseDataLoader &loader, BatchPrefetcher *prefetcher,
                                 size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
  if (prefetcher) {
    return prefetcher->next(batch_data, batch_labels);
  }
  return loader.get_batch(batch_size, batch_data, batch_labels);
}

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
  std::cout << "Async Pipeline: "
            << (config.async_pipeline ? "1 (async overlap)" : "0 (sync/no overlap)") << std::endl;

  std::unique_ptr<BatchPrefetcher> prefetcher;
  if (config.prefetch_data) {
    prefetcher =
        std::make_unique<BatchPrefetcher>(*train_loader, config.batch_size, config.prefetch_depth);
    prefetcher->start();
    std::cout << "Data prefetching enabled. Depth:" << config.prefetch_depth << std::endl;
  }

  while (get_next_train_batch(*train_loader, prefetcher.get(), config.batch_size, batch_data,
                              batch_labels)) {
    // Split batch into micro-batches
    Vec<Tensor> micro_batch_inputs = batch_data->split(0, config.num_microbatches);
    Vec<Tensor> micro_batch_labels = batch_labels->split(0, config.num_microbatches);

    auto process_start = std::chrono::high_resolution_clock::now();
    // Select pipeline schedule. Async overlaps microbatches; sync disables overlap.
    auto [loss, corrects] =
        coordinator.async_train_batch(micro_batch_inputs, micro_batch_labels, criterion);

    double ppl = std::exp(static_cast<double>(loss));

    size_t class_samples = 1;
    for (size_t i = 0; i < batch_labels->dims(); ++i) {
      class_samples *= static_cast<size_t>(batch_labels->shape()[i]);
    }

    total_loss += loss;
    total_corrects += corrects;
    total_samples += class_samples;
    accumulation_steps++;
    if (accumulation_steps == config.gradient_accumulation_steps) {
      coordinator.update_parameters();
      accumulation_steps = 0;
    }

    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

    // Log batch metrics to CSV.
    {
      long time_ms = process_duration.count() / 1000;  // us -> ms
      double acc_pct =
          total_samples > 0 ? static_cast<double>(total_corrects) / total_samples * 100.0 : 0.0;

      std::unordered_map<std::string, double> metrics;
      if (config.log_mode.log_loss) {
        // Keep legacy "loss" as the per-batch loss for backward compatibility.
        const double batch_loss = static_cast<double>(loss);
        const double avg_loss = (batch_index + 1) > 0 ? static_cast<double>(total_loss) /
                                                            static_cast<double>(batch_index + 1)
                                                      : batch_loss;
        metrics["loss"] = batch_loss;
        metrics["batch_loss"] = batch_loss;
        metrics["avg_loss"] = avg_loss;
        metrics["perplexity"] = std::exp(batch_loss);
        metrics["avg_perplexity"] = std::exp(avg_loss);
      }
      if (config.log_mode.log_accuracy) {
        metrics["accuracy_pct"] = acc_pct;
      }
      metrics["time_ms"] = time_ms;

      logger.log_batch(epoch, static_cast<int>(batch_index + 1), metrics);
    }

    if ((batch_index + 1) % config.progress_print_interval == 0) {
      std::cout << "Batch " << (batch_index + 1) << " Loss: " << std::fixed << std::setprecision(5)
                << loss << ", PPL: " << std::setprecision(2) << ppl << ", Cummulative Accuracy: "
                << (static_cast<double>(total_corrects) / total_samples * 100.0f) << "%"
                << ", Processing Time: " << process_duration.count() << " us" << std::endl;
    }
    ++batch_index;
  }

  auto epoch_end = std::chrono::high_resolution_clock::now();
  auto epoch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
  const float avg_train_loss =
      batch_index > 0 ? total_loss / static_cast<float>(batch_index) : 0.0f;
  const float train_accuracy_pct =
      total_samples > 0 ? static_cast<float>(static_cast<double>(total_corrects) /
                                             static_cast<double>(total_samples) * 100.0)
                        : 0.0f;

  std::cout << "\nEpoch completed in " << epoch_duration.count() << " milliseconds"
            << " | Train Loss: " << avg_train_loss << " | Train Accuracy: " << train_accuracy_pct
            << "%" << std::endl;

  return {avg_train_loss, train_accuracy_pct};
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
        config.async_pipeline
            ? coordinator.async_val_batch(micro_batch_inputs, micro_batch_labels, criterion)
            : coordinator.sync_val_batch(micro_batch_inputs, micro_batch_labels, criterion);
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

inline void train_semi_async_step(Coordinator &coordinator,
                                  std::unique_ptr<BaseDataLoader> &train_loader,
                                  const std::unique_ptr<Loss> &criterion,
                                  const TrainingConfig &config, CsvLogger &logger) {
  train_loader->shuffle();
  train_loader->reset();

  Tensor batch_data, batch_labels;
  int accumulation_steps = 0;

  double running_loss_sum = 0.0;
  size_t running_loss_items = 0;

  auto train_start = std::chrono::high_resolution_clock::now();

  std::cout << "Starting training for " << config.max_steps << " steps..." << std::endl;
  coordinator.set_training(true);

  std::unique_ptr<BatchPrefetcher> prefetcher;
  auto start_prefetcher = [&]() {
    prefetcher.reset();
    if (config.prefetch_data) {
      prefetcher = std::make_unique<BatchPrefetcher>(*train_loader, config.batch_size,
                                                     config.prefetch_depth);
      prefetcher->start();
    }
  };
  start_prefetcher();

  for (int step = 0; step < config.max_steps; ++step) {
    if (!get_next_train_batch(*train_loader, prefetcher.get(), config.batch_size, batch_data,
                              batch_labels)) {
      if (prefetcher) {
        prefetcher->stop();
      }
      train_loader->shuffle();
      train_loader->reset();
      start_prefetcher();
      get_next_train_batch(*train_loader, prefetcher.get(), config.batch_size, batch_data,
                           batch_labels);
    }

    // Split batch into micro-batches
    Vec<Tensor> micro_batch_inputs;
    ops::split(batch_data, micro_batch_inputs, config.num_microbatches);
    Vec<Tensor> micro_batch_labels;
    ops::split(batch_labels, micro_batch_labels, config.num_microbatches);

    auto process_start = std::chrono::high_resolution_clock::now();
    // Select pipeline schedule. Async overlaps microbatches; sync disables overlap.
    auto [loss, corrects] =
        coordinator.async_train_batch(micro_batch_inputs, micro_batch_labels, criterion);
    double ppl = std::exp(static_cast<double>(loss));

    size_t class_samples = 1;
    for (size_t i = 0; i < batch_labels->dims(); ++i) {
      class_samples *= static_cast<size_t>(batch_labels->shape()[i]);
    }

    running_loss_sum += static_cast<double>(loss) * static_cast<double>(class_samples);
    running_loss_items += class_samples;
    double avg_loss = (running_loss_items > 0)
                          ? running_loss_sum / static_cast<double>(running_loss_items)
                          : static_cast<double>(loss);
    double avg_ppl = std::exp(avg_loss);

    accumulation_steps++;
    if (accumulation_steps == config.gradient_accumulation_steps) {
      coordinator.update_parameters();
      accumulation_steps = 0;
    }

    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

    // Log batch metrics to CSV.
    {
      long time_ms = process_duration.count() / 1000;  // us -> ms
      double acc_pct =
          class_samples > 0 ? static_cast<double>(corrects) / class_samples * 100.0 : 0.0;

      std::unordered_map<std::string, double> metrics;
      if (config.log_mode.log_loss) {
        const double batch_loss = static_cast<double>(loss);

        metrics["loss"] = batch_loss;
        metrics["batch_loss"] = batch_loss;
        metrics["avg_loss"] = avg_loss;

        metrics["perplexity"] = ppl;
        metrics["avg_perplexity"] = avg_ppl;
      }

      if (config.log_mode.log_accuracy) {
        metrics["accuracy_pct"] = acc_pct;
      }

      metrics["time_ms"] = time_ms;

      logger.log_batch(1, step + 1, metrics);
    }

    if ((step + 1) % config.progress_print_interval == 0) {
      std::cout << "Step " << (step + 1) << " BatchLoss: " << std::fixed << std::setprecision(5)
                << loss << ", AvgLoss: " << std::fixed << std::setprecision(5) << avg_loss
                << ", PPL: " << std::setprecision(2) << ppl << ", AvgPPL: " << std::setprecision(2)
                << avg_ppl << ", Accuracy: " << std::setprecision(2)
                << (static_cast<double>(corrects) / class_samples * 100.0f) << "%"
                << ", Processing Time: " << process_duration.count() << " us" << std::endl;
    }
  }

  auto train_end = std::chrono::high_resolution_clock::now();
  auto train_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
  std::cout << "\nTraining completed in " << train_duration.count() << " milliseconds" << std::endl;
}

inline void train_model(Coordinator &coordinator, std::unique_ptr<BaseDataLoader> &train_loader,
                        std::unique_ptr<BaseDataLoader> &val_loader,
                        const std::unique_ptr<Loss> &criterion,
                        TrainingConfig config = TrainingConfig()) {
  coordinator.start_profiling();
  ThreadWrapper thread_wrapper({config.num_threads});
  CsvLogger logger("tnn_" + config.model_name, config.log_dir, &config.log_mode);

  const bool use_epoch_mode =
      (config.train_mode == "epoch") || (config.train_mode == "auto" && config.max_steps == -1);

  thread_wrapper.execute([&]() -> void {
    if (use_epoch_mode) {
      // Full epoch mode: train through the whole loader, then validation and epoch CSV summary.
      for (int epoch = 0; epoch < config.epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs << " ===" << std::endl;

        auto epoch_total_start = std::chrono::high_resolution_clock::now();

        auto train_start = std::chrono::high_resolution_clock::now();
        auto [train_loss, train_acc] =
            train_semi_async_epoch(coordinator, train_loader, criterion, config, logger, epoch + 1);
        auto train_end = std::chrono::high_resolution_clock::now();

        auto val_start = std::chrono::high_resolution_clock::now();
        auto [val_loss, val_acc] =
            validate_semi_async_epoch(coordinator, val_loader, criterion, config);
        auto val_end = std::chrono::high_resolution_clock::now();

        auto epoch_total_end = std::chrono::high_resolution_clock::now();

        const auto train_time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
        const auto val_time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(val_end - val_start).count();
        const auto epoch_total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             epoch_total_end - epoch_total_start)
                                             .count();

        std::cout << "Epoch timing | train_time=" << train_time_ms
                  << " ms | val_time=" << val_time_ms << " ms | total_time=" << epoch_total_time_ms
                  << " ms" << std::endl;

        std::unordered_map<std::string, double> metrics;
        if (config.log_mode.log_loss) {
          metrics["train_loss"] = train_loss;
          metrics["val_loss"] = val_loss;
        }
        if (config.log_mode.log_accuracy) {
          metrics["train_accuracy_pct"] = train_acc;
          metrics["val_accuracy_pct"] = val_acc;
        }

        metrics["train_time_ms"] = static_cast<double>(train_time_ms);
        metrics["val_time_ms"] = static_cast<double>(val_time_ms);
        metrics["epoch_total_time_ms"] = static_cast<double>(epoch_total_time_ms);

        logger.log_epoch(epoch + 1, metrics);
      }
    } else {
      // Fixed-step/batch mode: no validation/epoch summary; useful for GPT/OpenWebText runs.
      TrainingConfig batch_config = config;
      if (batch_config.max_steps <= 0) {
        batch_config.max_steps = static_cast<int64_t>(train_loader->size());
      }
      train_semi_async_step(coordinator, train_loader, criterion, batch_config, logger);
    }

    coordinator.fetch_profiling();
    coordinator.print_logs();
    logger.flush();
  });
}

}  // namespace tnn