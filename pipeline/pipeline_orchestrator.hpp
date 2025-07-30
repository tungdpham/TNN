#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <limits>
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

        std::vector<std::future<Tensor<T>>> forward_futures;
        std::vector<std::future<Tensor<T>>> backward_futures;

        // --- 1. Forward Pass ---
        for (int i = 0; i < num_micro_batches_; ++i) {
            Tensor<T> current_input = micro_batch_inputs[i];
            for (size_t j = 0; j < stages_.size(); ++j) {
                // This is a simplified, synchronous view of communication.
                // A real implementation would be more asynchronous.
                if (j > 0) {
                    current_input = communicator_->receive(j - 1);
                }
                auto future = stages_[j]->forward(current_input, i); // Pass micro-batch ID
                if (j < stages_.size() - 1) {
                    communicator_->send(future.get(), j + 1);
                } else {
                    forward_futures.push_back(std::move(future));
                }
            }
        }

        // --- Calculate Loss and Accuracy ---
        float total_loss = 0.0f;
        float total_correct = 0.0f;
        std::vector<Tensor<T>> predictions;
        predictions.reserve(num_micro_batches_);

        for (int i = 0; i < num_micro_batches_; ++i) {
            Tensor<T> micro_batch_prediction = forward_futures[i].get();
            total_loss += TensorCrossEntropyLoss<T>::compute_loss(micro_batch_prediction, micro_batch_targets[i]);
            total_correct += calculate_tensor_accuracy<T>(micro_batch_prediction, micro_batch_targets[i]);
            predictions.push_back(std::move(micro_batch_prediction));
        }


        // --- 2. Backward Pass ---
        for (int i = 0; i < num_micro_batches_; ++i) {
            // The loss would be computed here, and the initial gradient calculated.
            // For simplicity, let's assume a dummy gradient.
            Tensor<T> grad = compute_loss_gradient(predictions[i], micro_batch_targets[i]);

            for (int j = stages_.size() - 1; j >= 0; --j) {
                if (j < stages_.size() - 1) {
                    grad = communicator_->receive(j + 1);
                }
                auto future = stages_[j]->backward(grad, i); // Pass micro-batch ID
                if (j > 0) {
                    communicator_->send(future.get(), j - 1);
                } else {
                    backward_futures.push_back(std::move(future));
                }
            }
        }

        // Wait for all backward passes to complete
        for(auto& future : backward_futures) {
            future.get();
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
        return prediction - target;
    }

    std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
    std::unique_ptr<Communicator<T>> communicator_;
    int num_micro_batches_;
};

} // namespace pipeline
