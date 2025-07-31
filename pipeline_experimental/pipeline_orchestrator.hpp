#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <limits>
#include <thread>
#include "pipeline_stage.hpp"
#include "../tensor/tensor.hpp"

namespace pipeline {

// --- Utility functions moved here for orchestrator use ---

template <typename T>
class TensorCrossEntropyLoss {
public:
  // Returns the sum of losses for the batch, not the average.
  static float compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) {
    float total_loss = 0.0;
    const T epsilon = 1e-15;
    for (size_t i = 0; i < predictions.batch_size(); ++i) {
      for (size_t j = 0; j < predictions.channels(); ++j) {
        if (targets(i, j, 0, 0) > 0.5) {
          total_loss -= std::log(std::max(epsilon, predictions(i, j, 0, 0)));
        }
      }
    }
    return total_loss;
  }

  static Tensor<T> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets) {
    Tensor<T> gradient = predictions;

    size_t batch_size = predictions.batch_size();
    size_t num_classes = predictions.channels();

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        gradient(i, j, 0, 0) = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
      }
    }

    // Average over batch
    gradient /= static_cast<T>(batch_size);
    return gradient;
  }
};

template <typename T>
// Returns the count of correct predictions, not the accuracy.
float calculate_tensor_accuracy(const Tensor<T> &predictions, const Tensor<T> &targets) {
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
        if (pred_class == true_class) correct++;
    }
    return static_cast<float>(correct);
}


template <typename T>
class PipelineOrchestrator {
public:
    PipelineOrchestrator(std::vector<std::unique_ptr<PipelineStage<T>>> stages,
                         std::unique_ptr<Communicator<T>> communicator,
                         int num_micro_batches)
        : stages_(std::move(stages)),
          communicator_(std::move(communicator)),
          num_micro_batches_(num_micro_batches) {}

    std::pair<float, float> train_batch(const Tensor<T>& input_batch, const Tensor<T>& target_batch) {
        // Split the batch into micro-batches
        std::vector<Tensor<T>> micro_batch_inputs = input_batch.split(num_micro_batches_);
        std::vector<Tensor<T>> micro_batch_targets = target_batch.split(num_micro_batches_);

        std::vector<std::thread> micro_batch_threads;
        std::vector<Tensor<T>> predictions(num_micro_batches_);
        std::mutex predictions_mutex;

        for (int i = 0; i < num_micro_batches_; ++i) {
            micro_batch_threads.emplace_back([this, i, &micro_batch_inputs, &micro_batch_targets, &predictions, &predictions_mutex]() {
                // Forward pass
                Tensor<T> current_tensor = micro_batch_inputs[i];
                for (size_t j = 0; j < stages_.size(); ++j) {
                    auto future = stages_[j]->forward(current_tensor, i);
                    if (j < stages_.size() - 1) {
                        // This is now non-blocking
                        communicator_->send(std::move(future), j, i);
                        current_tensor = communicator_->receive(j, i);
                    } else {
                        // Last stage
                        current_tensor = future.get();
                    }
                }
                
                {
                    std::lock_guard<std::mutex> lock(predictions_mutex);
                    predictions[i] = current_tensor;
                }

                // Backward pass
                Tensor<T> grad = compute_loss_gradient(current_tensor, micro_batch_targets[i]);
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
            });
        }

        for (auto& t : micro_batch_threads) {
            t.join();
        }

        // --- Calculate metrics ---
        float total_loss = 0.0f;
        float total_correct = 0.0f;

        for (int i = 0; i < num_micro_batches_; ++i) {
            total_loss += TensorCrossEntropyLoss<T>::compute_loss(predictions[i], micro_batch_targets[i]);
            total_correct += calculate_tensor_accuracy<T>(predictions[i], micro_batch_targets[i]);
        }

        return {total_loss / input_batch.batch_size(), total_correct / input_batch.batch_size()};
    }

    const std::vector<std::unique_ptr<PipelineStage<T>>>& get_stages() const {
        return stages_;
    }
private:
    Tensor<T> compute_loss_gradient(const Tensor<T>& prediction, const Tensor<T>& target) {
        // This is where your loss function's backward pass would be called.
        // For example, if using Mean Squared Error, the gradient is (prediction - target).
        return TensorCrossEntropyLoss<T>::compute_gradient(prediction, target);
    }

    std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
    std::unique_ptr<Communicator<T>> communicator_;
    int num_micro_batches_;
};

} // namespace pipeline
