#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>
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
        auto train_start_time = std::chrono::high_resolution_clock::now();
        // Split the batch into micro-batches
        std::vector<Tensor<T>> micro_batch_inputs = input_batch.split(num_micro_batches_);
        std::vector<Tensor<T>> micro_batch_targets = target_batch.split(num_micro_batches_);

        std::vector<std::future<Tensor<T>>> forward_futures(num_micro_batches_);
        std::vector<std::future<Tensor<T>>> backward_futures(num_micro_batches_);
        std::vector<Tensor<T>> predictions(num_micro_batches_);

        // --- Asynchronous Pipeline Execution ---
        auto warmup_start_time = std::chrono::high_resolution_clock::now();
        // 1. Warm-up: Fill the pipeline with forward passes
        for (int i = 0; i < stages_.size() - 1; ++i) {
            Tensor<T> current_input = micro_batch_inputs[i];
            for (size_t j = 0; j < stages_.size(); ++j) {
                if (j > 0) {
                    current_input = communicator_->receive(j - 1, i);
                }
                auto future = stages_[j]->forward(current_input, i);
                if (j < stages_.size() - 1) {
                    communicator_->send(std::move(future), j, i);
                } else {
                    forward_futures[i] = std::move(future);
                }
            }
        }
        auto warmup_end_time = std::chrono::high_resolution_clock::now();

        auto steady_state_start_time = std::chrono::high_resolution_clock::now();
        // 2. Steady State: 1F1B (one forward, one backward pass)
        for (int i = 0; i < num_micro_batches_ - (stages_.size() - 1); ++i) {
            int forward_batch_idx = i + stages_.size() - 1;
            int backward_batch_idx = i;

            // Forward pass for micro-batch `forward_batch_idx`
            Tensor<T> current_input = micro_batch_inputs[forward_batch_idx];
            for (size_t j = 0; j < stages_.size(); ++j) {
                if (j > 0) {
                    current_input = communicator_->receive(j - 1, forward_batch_idx);
                }
                auto future = stages_[j]->forward(current_input, forward_batch_idx);
                if (j < stages_.size() - 1) {
                    communicator_->send(std::move(future), j, forward_batch_idx);
                } else {
                    forward_futures[forward_batch_idx] = std::move(future);
                }
            }

            // Backward pass for micro-batch `backward_batch_idx`
            predictions[backward_batch_idx] = forward_futures[backward_batch_idx].get();
            Tensor<T> grad = compute_loss_gradient(predictions[backward_batch_idx], micro_batch_targets[backward_batch_idx]);
            for (int j = stages_.size() - 1; j >= 0; --j) {
                if (j < stages_.size() - 1) {
                    grad = communicator_->receive_grad(j, backward_batch_idx);
                }
                auto future = stages_[j]->backward(grad, backward_batch_idx);
                if (j > 0) {
                    communicator_->send_grad(std::move(future), j - 1, backward_batch_idx);
                } else {
                    backward_futures[backward_batch_idx] = std::move(future);
                }
            }
        }
        auto steady_state_end_time = std::chrono::high_resolution_clock::now();

        auto cooldown_start_time = std::chrono::high_resolution_clock::now();
        // 3. Cool-down: Drain the pipeline with backward passes
        for (int i = num_micro_batches_ - (stages_.size() - 1); i < num_micro_batches_; ++i) {
            int backward_batch_idx = i;
            predictions[backward_batch_idx] = forward_futures[backward_batch_idx].get();
            Tensor<T> grad = compute_loss_gradient(predictions[backward_batch_idx], micro_batch_targets[backward_batch_idx]);
            for (int j = stages_.size() - 1; j >= 0; --j) {
                if (j < stages_.size() - 1) {
                    grad = communicator_->receive_grad(j, backward_batch_idx);
                }
                auto future = stages_[j]->backward(grad, backward_batch_idx);
                if (j > 0) {
                    communicator_->send_grad(std::move(future), j - 1, backward_batch_idx);
                } else {
                    backward_futures[backward_batch_idx] = std::move(future);
                }
            }
        }
        auto cooldown_end_time = std::chrono::high_resolution_clock::now();

        // --- Wait for all backward passes to complete and calculate metrics ---
        auto wait_start_time = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        float total_correct = 0.0f;

        for (int i = 0; i < num_micro_batches_; ++i) {
            backward_futures[i].get(); // Ensure backward pass is complete
            total_loss += TensorCrossEntropyLoss<T>::compute_loss(predictions[i], micro_batch_targets[i]);
            total_correct += calculate_tensor_accuracy<T>(predictions[i], micro_batch_targets[i]);
        }
        auto wait_end_time = std::chrono::high_resolution_clock::now();

        // --- Apply Gradients ---
        for (auto& stage : stages_) {
            stage->apply_gradients(num_micro_batches_);
        }

        auto train_end_time = std::chrono::high_resolution_clock::now();

        // Store profiling data
        warmup_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end_time - warmup_start_time).count();
        steady_state_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(steady_state_end_time - steady_state_start_time).count();
        cooldown_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(cooldown_end_time - cooldown_start_time).count();
        wait_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(wait_end_time - wait_start_time).count();
        total_train_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(train_end_time - train_start_time).count();

        return {total_loss / input_batch.batch_size(), total_correct / input_batch.batch_size()};
    }

    void print_profiling_info() const {
        std::cout << "--- Pipeline Orchestrator Profiling ---" << std::endl;
        std::cout << "Total Training Time: " << std::fixed << std::setprecision(2) << total_train_time_ms_ / 1000.0 << " s" << std::endl;
        std::cout << "  Warm-up Phase:     " << std::fixed << std::setprecision(2) << warmup_time_ms_ << " ms" << std::endl;
        std::cout << "  Steady State Phase:  " << std::fixed << std::setprecision(2) << steady_state_time_ms_ << " ms" << std::endl;
        std::cout << "  Cool-down Phase:   " << std::fixed << std::setprecision(2) << cooldown_time_ms_ << " ms" << std::endl;
        std::cout << "  Result Wait Time:    " << std::fixed << std::setprecision(2) << wait_time_ms_ << " ms" << std::endl;
        std::cout << "---------------------------------------" << std::endl;

        for (const auto& stage : stages_) {
            stage->print_profiling_info();
        }
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
    // Profiling data
    long long warmup_time_ms_ = 0;
    long long steady_state_time_ms_ = 0;
    long long cooldown_time_ms_ = 0;
    long long wait_time_ms_ = 0;
    long long total_train_time_ms_ = 0;
};

} // namespace pipeline
