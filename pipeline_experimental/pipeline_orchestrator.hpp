#pragma once

#include "../nn/loss.hpp"
#include "../tensor/tensor.hpp"
#include "pipeline_stage.hpp"
#include "thread_pool.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace pipeline {

template <typename T>
// Returns the count of correct predictions, not the accuracy.
float calculate_tensor_accuracy(const Tensor<T> &predictions,
                                const Tensor<T> &targets) {
  int correct = 0;
  for (size_t i = 0; i < predictions.batch_size(); ++i) {
    int pred_class = 0, true_class = 0;
    T max_pred = -1.0;
    for (size_t j = 0; j < predictions.channels(); ++j) {
      if (predictions(i, j, 0, 0) > max_pred) {
        max_pred = predictions(i, j, 0, 0);
        pred_class = j;
      }
      if (targets(i, j, 0, 0) > 0.5) {
        true_class = j;
      }
    }
    if (pred_class == true_class)
      correct++;
  }
  return static_cast<float>(correct);
}

template <typename T> class PipelineOrchestrator {
public:
  PipelineOrchestrator(std::vector<std::unique_ptr<PipelineStage<T>>> stages,
                       std::unique_ptr<Communicator<T>> communicator,
                       int num_micro_batches, size_t num_threads = 4)
      : stages_(std::move(stages)), communicator_(std::move(communicator)),
        num_micro_batches_(num_micro_batches), thread_pool_(num_threads) {
    loss_function_ = layers::LossFactory<float>::create_crossentropy(1e-15);
  }

  std::pair<float, float> train_batch(const Tensor<T> &input_batch,
                                      const Tensor<T> &target_batch) {
    // Split the batch into micro-batches
    std::vector<Tensor<T>> micro_batch_inputs =
        input_batch.split(num_micro_batches_);
    std::vector<Tensor<T>> micro_batch_targets =
        target_batch.split(num_micro_batches_);

    std::vector<std::future<void>> micro_batch_futures;
    std::vector<Tensor<T>> predictions(num_micro_batches_);
    std::mutex predictions_mutex;
    
    for (int i = 0; i < num_micro_batches_; ++i) {
      micro_batch_futures.push_back(thread_pool_.enqueue(
          [this, i, &micro_batch_inputs, &micro_batch_targets, &predictions,
           &predictions_mutex]() {
            Tensor<T> current_tensor = micro_batch_inputs[i];
            for (size_t j = 0; j < stages_.size(); ++j) {
              auto future = stages_[j]->forward(current_tensor, i);
              if (j < stages_.size() - 1) {
                communicator_->send(std::move(future), j, i);
                current_tensor = communicator_->receive(j, i);
              } else {
                // Last stage
                current_tensor = future.get();
              }
            }

            // Store predictions
            {
              std::lock_guard<std::mutex> lock(predictions_mutex);
              predictions[i] = current_tensor;
            }

            // Backward pass
            Tensor<T> grad = loss_function_->compute_gradient(
                current_tensor, micro_batch_targets[i]);
            for (int j = stages_.size() - 1; j >= 0; --j) {
              auto future = stages_[j]->backward(grad, i);
              if (j > 0) {
                communicator_->send_grad(std::move(future), j - 1, i);
                grad = communicator_->receive_grad(j - 1, i);
              } else {
                // First stage, no one to send gradient to.
                future.get();
              }
            }

            // printf("Micro-batch %d backwarded\n", i);
          }));
    }

    for (auto &future : micro_batch_futures) {
      future.get();
    }

    // --- Calculate metrics ---
    float total_loss = 0.0f;
    float total_correct = 0.0f;

    for (int i = 0; i < num_micro_batches_; ++i) {
      total_loss +=
          loss_function_->compute_loss(predictions[i], micro_batch_targets[i]);
      total_correct +=
          calculate_tensor_accuracy<T>(predictions[i], micro_batch_targets[i]);
    }

    return {total_loss / input_batch.batch_size(),
            total_correct / input_batch.batch_size()};
  }

  const std::vector<std::unique_ptr<PipelineStage<T>>> &get_stages() const {
    return stages_;
  }

private:
  std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
  std::unique_ptr<Communicator<T>> communicator_;
  int num_micro_batches_;
  ThreadPool thread_pool_;
  std::unique_ptr<layers::Loss<float>> loss_function_;
};

} // namespace pipeline
