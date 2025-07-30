#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <map>

#include "../layers/layers.hpp"
#include "../tensor/tensor.hpp"
#include "communication.hpp"
#include "thread_pool.hpp"

namespace pipeline {

// A simple profiler to measure execution times
struct StageProfiler {
    std::chrono::high_resolution_clock::time_point forward_start;
    std::chrono::high_resolution_clock::time_point forward_end;
    std::chrono::high_resolution_clock::time_point backward_start;
    std::chrono::high_resolution_clock::time_point backward_end;
};

template <typename T>
class PipelineStage {
public:
    PipelineStage(int stage_id, int device_id, size_t num_threads = 1, int omp_num_threads = 1)
        : stage_id_(stage_id), device_id_(device_id), pool_(num_threads, omp_num_threads) {}

    void add_layer(std::unique_ptr<layers::Layer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    // Asynchronous forward pass for a micro-batch
    std::future<Tensor<T>> forward(Tensor<T> input, int micro_batch_id) {
        auto task = [this, input, micro_batch_id]() mutable {
            profilers_[micro_batch_id].forward_start = std::chrono::high_resolution_clock::now();
            Tensor<T> current_tensor = input;
            for (auto& layer : layers_) {
                current_tensor = layer->forward(current_tensor, micro_batch_id);
            }
            profilers_[micro_batch_id].forward_end = std::chrono::high_resolution_clock::now();
            return current_tensor;
        };
        return pool_.enqueue(std::move(task));
    }

    // Asynchronous backward pass for a micro-batch
    std::future<Tensor<T>> backward(Tensor<T> grad_output, int micro_batch_id) {
        auto task = [this, grad_output, micro_batch_id]() mutable {
            profilers_[micro_batch_id].backward_start = std::chrono::high_resolution_clock::now();
            Tensor<T> current_grad = grad_output;
            
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                current_grad = (*it)->backward(current_grad, micro_batch_id);
            }

            // After the backward pass for a micro-batch, gradients for the weights in this stage are computed.
            // These gradients need to be accumulated.
            accumulate_gradients();

            
            profilers_[micro_batch_id].backward_end = std::chrono::high_resolution_clock::now();

            return current_grad;
        };
        return pool_.enqueue(std::move(task));
    }

    void set_optimizer(std::unique_ptr<layers::Optimizer<T>> optimizer) {
        optimizer_ = std::move(optimizer);
    }

    void apply_gradients(int num_micro_batches) {
        if (!optimizer_) {
            // It's possible for a stage to have no parameters
            return;
        }

        std::lock_guard<std::mutex> lock(accumulation_mutex_);

        if (accumulated_gradients_.empty()) {
            return;
        }

        // Average the accumulated gradients and copy them to the layer's gradient tensors
        for (auto const& [layer_ptr, acc_grads] : accumulated_gradients_) {
            auto layer_grads_ptrs = layer_ptr->gradients();
            for (size_t i = 0; i < acc_grads.size(); ++i) {
                *layer_grads_ptrs[i] = acc_grads[i] / static_cast<T>(num_micro_batches);
            }
        }

        optimizer_->step(); // Update weights

        accumulated_gradients_.clear(); // Clear for next mini-batch
    }

    void print_profiling_info() const {
        long long total_forward_us = 0;
        long long total_backward_us = 0;
        for (const auto& pair : profilers_) {
            total_forward_us += std::chrono::duration_cast<std::chrono::microseconds>(pair.second.forward_end - pair.second.forward_start).count();
            total_backward_us += std::chrono::duration_cast<std::chrono::microseconds>(pair.second.backward_end - pair.second.backward_start).count();
        }
        long long avg_forward_us = profilers_.empty() ? 0 : total_forward_us / profilers_.size();
        long long avg_backward_us = profilers_.empty() ? 0 : total_backward_us / profilers_.size();

        std::cout << "  Stage " << stage_id_ << " Profiling:" << std::endl;
        std::cout << "    Average Forward Pass: " << std::fixed << std::setprecision(2) << avg_forward_us / 1000.0 << " ms" << std::endl;
        std::cout << "    Average Backward Pass: " << std::fixed << std::setprecision(2) << avg_backward_us / 1000.0 << " ms" << std::endl;
    }

    std::vector<Tensor<T>*> parameters() {
        std::vector<Tensor<T>*> params;
        for (auto& layer : layers_) {
            if (layer->has_parameters()) {
                auto layer_params = layer->parameters();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
        }
        return params;
    }

    std::vector<Tensor<T>*> gradients() {
        std::vector<Tensor<T>*> grads;
        for (auto& layer : layers_) {
            if (layer->has_parameters()) {
                auto layer_grads = layer->gradients();
                grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
            }
        }
        return grads;
    }

    void accumulate_gradients() {
        std::lock_guard<std::mutex> lock(accumulation_mutex_);
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (layers_[i]->has_parameters()) {
                auto* layer_params =
                    dynamic_cast<layers::ParameterizedLayer<T>*>(layers_[i].get());
                auto grads = layer_params->gradients();

                if (!accumulated_gradients_.count(layer_params)) {
                    // First time, so initialize accumulated gradients
                    std::vector<Tensor<T>> new_grads;
                    for (const auto& grad : grads) {
                        new_grads.push_back(grad->clone());
                    }
                    accumulated_gradients_[layer_params] = new_grads;
                } else {
                    // Accumulate
                    auto& acc_grads = accumulated_gradients_.at(layer_params);
                    for (size_t j = 0; j < grads.size(); ++j) {
                        if (acc_grads[j].shape() != grads[j]->shape()) {
                            // This check can help debug, but the mutex should prevent this.
                            throw std::runtime_error("Shape mismatch during accumulation!");
                        }
                        acc_grads[j] += *grads[j];
                    }
                }
            }
        }
    }

private:
    int stage_id_;
    int device_id_; // For future use with GPUs
    std::vector<std::unique_ptr<layers::Layer<T>>> layers_;
    ThreadPool pool_;
    std::unique_ptr<layers::Optimizer<T>> optimizer_;

    // Gradient accumulation
    std::map<layers::ParameterizedLayer<T>*, std::vector<Tensor<T>>> accumulated_gradients_;
    std::mutex accumulation_mutex_;
    std::map<int, StageProfiler> profilers_;
};

} // namespace pipeline
