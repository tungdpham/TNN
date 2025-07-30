#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "../layers/layers.hpp"
#include "../tensor/tensor.hpp"
#include "communication.hpp"
#include "thread_pool.hpp"

namespace pipeline {

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
            Tensor<T> current_tensor = input;
            for (auto& layer : layers_) {
                current_tensor = layer->forward(current_tensor, micro_batch_id);
            }
            return current_tensor;
        };
        return pool_.enqueue(std::move(task));
    }

    // Asynchronous backward pass for a micro-batch
    std::future<Tensor<T>> backward(Tensor<T> grad_output, int micro_batch_id) {
        auto task = [this, grad_output, micro_batch_id]() mutable {
            Tensor<T> current_grad = grad_output;
            
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                current_grad = (*it)->backward(current_grad, micro_batch_id);
            }

            // After the backward pass for a micro-batch, gradients for the weights in this stage are computed.
            // These gradients need to be accumulated.
            accumulate_gradients();

            return current_grad;
        };
        return pool_.enqueue(std::move(task));
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

    void apply_gradients(layers::Optimizer<T>& optimizer) {
        std::lock_guard<std::mutex> lock(gradient_mutex_);
        // This is a simplified application of gradients. A real implementation
        // might need to handle parameter-gradient mapping more carefully.
        auto params = parameters();
        // Debugging params
        // for (auto& param : params) {
        //     printf("Parameter shape: %s\n", param->shape_str().c_str());
        // }
        auto grads = gradients();

        // We need to create a temporary list of gradient *values* for the optimizer,
        // based on the accumulated gradients. This is a bit of a workaround given
        // the current optimizer API.
        std::vector<Tensor<T>> accumulated_grads_for_optim;
        for (auto& layer : layers_) {
            if(layer->has_parameters()){
                auto* p_layer = dynamic_cast<layers::ParameterizedLayer<T>*>(layer.get());
                if(accumulated_gradients_.count(p_layer)){
                    for(const auto& grad_tensor : accumulated_gradients_.at(p_layer)){
                        accumulated_grads_for_optim.push_back(grad_tensor);
                    }
                }
            }
        }
        // The optimizer needs pointers, so we create a vector of pointers to the accumulated gradients.
        std::vector<Tensor<T>*> grad_pointers;
        for(auto& grad : accumulated_grads_for_optim){
            grad_pointers.push_back(&grad);
        }

        if(!params.empty() && !grad_pointers.empty()){
             optimizer.update(params, grad_pointers);
        }

        accumulated_gradients_.clear();
    }


private:
    int stage_id_;
    int device_id_; // e.g., GPU id
    std::vector<std::unique_ptr<layers::Layer<T>>> layers_;
    ThreadPool pool_;

    // For gradient accumulation
    std::unordered_map<layers::ParameterizedLayer<T>*, std::vector<Tensor<T>>>
        accumulated_gradients_;
    std::mutex gradient_mutex_;
};

} // namespace pipeline
