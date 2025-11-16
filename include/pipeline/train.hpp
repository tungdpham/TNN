/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/train.hpp"
#include "pipeline/distributed_coordinator.hpp"

namespace tnn {

ClassResult train_semi_async_epoch(DistributedCoordinator &coordinator,
                                   BaseDataLoader<float> &train_loader,
                                   size_t progress_print_interval) {
  Tensor<float> batch_data, batch_labels;

  size_t batch_index = 0;

  auto epoch_start = std::chrono::high_resolution_clock::now();

  float total_loss = 0.0f;

  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    // Split batch into micro-batches
    std::vector<Tensor<float>> micro_batches = split(batch_data, coordinator.num_microbatches());
    std::vector<Tensor<float>> micro_batch_labels =
        split(batch_labels, coordinator.num_microbatches());

    auto process_start = std::chrono::high_resolution_clock::now();
    // Perform forward, compute loss, and backward asynchronously.
    total_loss += coordinator.async_process_batch(micro_batches, micro_batch_labels);
    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

    auto update_start = std::chrono::high_resolution_clock::now();
    coordinator.update_parameters();

    auto update_end = std::chrono::high_resolution_clock::now();
    auto update_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start);

    if ((batch_index + 1) % progress_print_interval == 0) {

      std::cout << "Async process completed in " << process_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Parameter update completed in " << update_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Average Loss after " << (batch_index + 1)
                << " batches: " << (total_loss / (batch_index + 1)) << std::endl;
      std::cout << "Batch " << batch_index + 1 << "/"
                << train_loader.size() / train_loader.get_batch_size() << std::endl;
    }
    ++batch_index;
  }

  auto epoch_end = std::chrono::high_resolution_clock::now();
  auto epoch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
  std::cout << "\nEpoch completed in " << epoch_duration.count() << " milliseconds" << std::endl;
  return {total_loss / batch_index, -1.0f};
}

ClassResult validate_semi_async_epoch(DistributedCoordinator &coordinator,
                                      BaseDataLoader<float> &test_loader) {
  Tensor<float> batch_data, batch_labels;

  float total_val_loss = 0.0f;
  float total_val_correct = 0.0f;
  int val_batches = 0;

  while (test_loader.get_next_batch(batch_data, batch_labels)) {
    std::vector<Tensor<float>> micro_batches = split(batch_data, coordinator.num_microbatches());

    std::vector<Tensor<float>> micro_batch_labels =
        split(batch_labels, coordinator.num_microbatches());

    for (size_t i = 0; i < micro_batches.size(); ++i) {
      coordinator.forward(micro_batches[i], i);
    }

    coordinator.join(CommandType::FORWARD_JOB, coordinator.num_microbatches(), 60);

    std::vector<Message> all_messages = coordinator.dequeue_all_messages(CommandType::FORWARD_JOB);

    if (all_messages.size() != static_cast<size_t>(coordinator.num_microbatches())) {
      throw std::runtime_error(
          "Unexpected number of messages: " + std::to_string(all_messages.size()) +
          ", expected: " + std::to_string(coordinator.num_microbatches()));
    }

    std::vector<Job<float>> forward_jobs;
    for (const auto &message : all_messages) {
      if (message.header.command_type == CommandType::FORWARD_JOB) {
        forward_jobs.push_back(message.get<Job<float>>());
      }
    }

    auto val_loss = 0.0f;
    auto val_correct = 0.0f;

    for (auto &job : forward_jobs) {
      val_loss += coordinator.compute_loss(job.data, micro_batch_labels[job.micro_batch_id]);
      val_correct +=
          compute_class_corrects<float>(job.data, micro_batch_labels[job.micro_batch_id]);
    }
    total_val_loss += val_loss;
    total_val_correct += val_correct;
    ++val_batches;
  }

  std::cout << "Validation completed." << std::endl;
  std::cout << "Average Validation Loss: " << (total_val_loss / val_batches)
            << ", Average Validation Accuracy: "
            << (total_val_correct / test_loader.size()) * 100.0f << "%" << std::endl;
  return {static_cast<float>(total_val_loss / val_batches),
          static_cast<float>((total_val_correct / test_loader.size()) * 100.0f)};
}

void train_model(DistributedCoordinator &coordinator, BaseDataLoader<float> &train_loader,
                 BaseDataLoader<float> &test_loader, TrainingConfig config = TrainingConfig()) {
  train_loader.prepare_batches(config.batch_size);
  test_loader.prepare_batches(config.batch_size);

  for (int epoch = 0; epoch < config.epochs; ++epoch) {
    std::cout << "\n=== Epoch " << (epoch + 1) << "/" << config.epochs << " ===" << std::endl;
    train_loader.reset();
    test_loader.reset();

    train_loader.shuffle();

    train_semi_async_epoch(coordinator, train_loader, config.progress_print_interval);

    validate_semi_async_epoch(coordinator, test_loader);

    train_loader.prepare_batches(config.batch_size);
  }
}

} // namespace tnn