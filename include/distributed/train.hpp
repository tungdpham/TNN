/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "coordinator.hpp"
#include "nn/accuracy.hpp"
#include "nn/train.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "threading/thread_wrapper.hpp"
#include <memory>

namespace tnn {

inline Result train_semi_async_epoch(Coordinator &coordinator, BaseDataLoader &train_loader,
                                     const std::unique_ptr<Loss> &criterion,
                                     const TrainingConfig &config) {
  Tensor batch_data, batch_labels;

  size_t batch_index = 0;

  auto epoch_start = std::chrono::high_resolution_clock::now();

  float total_loss = 0.0f;

  coordinator.set_training(true);

  while (train_loader.get_batch(config.batch_size, batch_data, batch_labels)) {
    // Split batch into micro-batches
    std::vector<Tensor> micro_batch_inputs;
    DISPATCH_AUTO_T(TensorOps::split, batch_data, micro_batch_inputs,
                    coordinator.num_microbatches());
    std::vector<Tensor> micro_batch_labels;
    DISPATCH_AUTO_T(TensorOps::split, batch_labels, micro_batch_labels,
                    coordinator.num_microbatches());

    auto process_start = std::chrono::high_resolution_clock::now();
    // Perform forward, compute loss, and backward asynchronously.
    total_loss +=
        coordinator.async_process_batch(micro_batch_inputs, micro_batch_labels, criterion);
    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

    coordinator.update_parameters();

    if ((batch_index + 1) % config.progress_print_interval == 0) {
      std::cout << "Async process completed in " << process_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Average Loss after " << (batch_index + 1)
                << " batches: " << (total_loss / (batch_index + 1)) << std::endl;
      std::cout << "Batch " << batch_index + 1 << "/" << train_loader.size() / config.batch_size
                << std::endl;
      if (config.profiler_type != ProfilerType::NONE) {
        coordinator.print_profiling();
      }
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

inline Result validate_semi_async_epoch(Coordinator &coordinator, BaseDataLoader &test_loader,
                                        const std::unique_ptr<Loss> &criterion,
                                        const TrainingConfig &config) {
  Tensor batch_data, batch_labels;
  float total_val_loss = 0.0f;
  float total_val_correct = 0.0f;
  int val_batches = 0;

  coordinator.set_training(false);

  while (test_loader.get_batch(config.batch_size, batch_data, batch_labels)) {
    std::vector<Tensor> micro_batch_inputs;
    DISPATCH_AUTO_T(TensorOps::split, batch_data, micro_batch_inputs,
                    coordinator.num_microbatches());

    std::vector<Tensor> micro_batch_labels;
    DISPATCH_AUTO_T(TensorOps::split, batch_labels, micro_batch_labels,
                    coordinator.num_microbatches());

    for (size_t i = 0; i < micro_batch_inputs.size(); ++i) {
      coordinator.forward(std::move(micro_batch_inputs[i]), i);
    }

    coordinator.join(CommandType::FORWARD_JOB, coordinator.num_microbatches(), 60);

    std::vector<Message> all_messages = coordinator.dequeue_all_messages(CommandType::FORWARD_JOB);

    if (all_messages.size() != static_cast<size_t>(coordinator.num_microbatches())) {
      throw std::runtime_error(
          "Unexpected number of messages: " + std::to_string(all_messages.size()) +
          ", expected: " + std::to_string(coordinator.num_microbatches()));
    }

    std::vector<Job *> forward_jobs;
    for (auto &message : all_messages) {
      if (message.header().command_type == CommandType::FORWARD_JOB) {
        forward_jobs.push_back(&message.get<Job>());
      }
    }

    double val_loss = 0.0;
    double val_correct = 0.0;

    for (auto &job : forward_jobs) {
      float loss = 0.0f;
      criterion->compute_loss(job->data, micro_batch_labels[job->micro_batch_id], loss);
      val_loss += loss;
      val_correct += compute_class_corrects(job->data, micro_batch_labels[job->micro_batch_id]);
    }
    // Normalize loss by number of microbatches to match training loss semantics
    if (coordinator.num_microbatches() > 0) {
      val_loss /= static_cast<double>(coordinator.num_microbatches());
    }
    total_val_loss += val_loss;
    total_val_correct += val_correct;
    ++val_batches;
  }

  std::cout << "Validation completed." << std::endl;
  std::cout << "Average Validation Loss: " << (total_val_loss / val_batches)
            << ", Average Validation Accuracy: "
            << (total_val_correct / test_loader.size()) * 100.0f << "%" << std::endl;
  return {total_val_loss / val_batches, (total_val_correct / test_loader.size()) * 100.0f};
}

inline void train_model(Coordinator &coordinator, BaseDataLoader &train_loader,
                        BaseDataLoader &test_loader, const std::unique_ptr<Loss> &criterion,
                        TrainingConfig config = TrainingConfig()) {
  coordinator.start_profiling();
  ThreadWrapper thread_wrapper({config.num_threads});
  coordinator.set_num_microbatches(config.num_microbatches);

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs << " ===" << std::endl;
      train_loader.reset();
      test_loader.reset();
      train_loader.shuffle();

      train_semi_async_epoch(coordinator, train_loader, criterion, config);

      validate_semi_async_epoch(coordinator, test_loader, criterion, config);

      coordinator.fetch_profiling();
    }
  });
}

} // namespace tnn