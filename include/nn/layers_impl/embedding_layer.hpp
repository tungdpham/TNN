/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/cpu/embedding_ops.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/embedding_ops.hpp"
#endif

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class EmbeddingLayer : public ParameterizedLayer<T> {
private:
  size_t vocab_size_;
  size_t embed_dim_;
  Tensor<T> weight_;
  Tensor<T> grad_weight_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;

  std::unique_ptr<Task> compute_forward_task(const device_ptr<T[]> &input,
                                             const device_ptr<T[]> &weight, device_ptr<T[]> &output,
                                             size_t num_indices, size_t vocab_size,
                                             size_t embed_dim, size_t padding_idx,
                                             const std::string &flow_id) const {
    if (input.device_type() == DeviceType::CPU) {
      return create_cpu_task(flow_id, cpu::embedding::compute_embedding_forward<T>, input.get(),
                             weight.get(), output.get(), num_indices, vocab_size, embed_dim,
                             padding_idx);
    }
#ifdef USE_CUDA
    else if (input.device_type() == DeviceType::GPU) {
      return create_gpu_task(flow_id, cuda::embedding::compute_embedding_forward<T>, input.get(),
                             weight.get(), output.get(), num_indices, vocab_size, embed_dim,
                             padding_idx);
    }
#endif
    else {
      throw std::runtime_error("Unsupported device type for embedding forward");
    }
  }

  std::unique_ptr<Task>
  compute_backward_task(const device_ptr<T[]> &input, const device_ptr<T[]> &gradient,
                        device_ptr<T[]> &grad_weight, size_t num_indices, size_t vocab_size,
                        size_t embed_dim, size_t padding_idx, const std::string &flow_id) const {
    if (input.device_type() == DeviceType::CPU) {
      return create_cpu_task(flow_id, cpu::embedding::compute_embedding_backward<T>, input.get(),
                             gradient.get(), grad_weight.get(), num_indices, vocab_size, embed_dim,
                             padding_idx);
    }
#ifdef USE_CUDA
    else if (input.device_type() == DeviceType::GPU) {
      return create_gpu_task(flow_id, cuda::embedding::compute_embedding_backward<T>, input.get(),
                             gradient.get(), grad_weight.get(), num_indices, vocab_size, embed_dim,
                             padding_idx);
    }
#endif
    else {
      throw std::runtime_error("Unsupported device type for embedding backward");
    }
  }

public:
  EmbeddingLayer(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding")
      : ParameterizedLayer<T>(name), vocab_size_(vocab_size), embed_dim_(embed_dim) {}

  void initialize_params() override {
    // weight shape: [vocab_size, embed_dim, 1, 1]
    weight_ = Tensor<T>({vocab_size_, embed_dim_, 1, 1}, this->device_);
    grad_weight_ = Tensor<T>({vocab_size_, embed_dim_, 1, 1}, this->device_);

    if (this->use_seed_) {
      weight_.fill_random_normal(0.0, 0.02, this->srand_seed_);
    } else {
      weight_.fill_random_normal(0.0, 0.02);
    }
    grad_weight_.fill(T(0));
  }

  void set_training(bool training) override { this->is_training_ = training; }

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    const Tensor<T> *current = &input;
    Tensor<T> device_input;
    if (input.device() != this->device_) {
      device_input = input.to_device(this->device_);
      current = &device_input;
    }

    if (this->is_training_) {
      micro_batch_inputs_[micro_batch_id] = current->clone();
    }

    size_t num_tokens = input.size();
    if (num_tokens == 0)
      return;

    std::vector<size_t> out_shape = input.shape();
    if (out_shape.size() >= 2) {
      if (out_shape.size() == 4 && out_shape[1] == 1) {
        out_shape[1] = embed_dim_;
      } else {
        throw std::runtime_error("EmbeddingLayer::forward: Input tensor must have shape [N, 1, H, "
                                 "W] or [N, L] where L is "
                                 "the number of tokens.");
      }
    }
    output.ensure(out_shape, this->device_);

    compute_forward_task(current->data_ptr(), weight_.data_ptr(), output.data_ptr(), num_tokens,
                         vocab_size_, embed_dim_, vocab_size_, "default");
  }

  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override {
    auto it = micro_batch_inputs_.find(micro_batch_id);
    if (it == micro_batch_inputs_.end()) {
      throw std::runtime_error("EmbeddingLayer::backward: No cached input for micro_batch_id " +
                               std::to_string(micro_batch_id));
    }
    const Tensor<T> *current_gradient = &gradient;
    Tensor<T> device_gradient;
    if (gradient.device() != this->device_) {
      device_gradient = gradient.to_device(this->device_);
      current_gradient = &device_gradient;
    }

    const Tensor<T> &input = it->second;

    grad_input.ensure(input.shape(), this->device_);
    grad_input.fill(T(0));

    size_t num_tokens = input.size();

    compute_backward_task(input.data_ptr(), current_gradient->data_ptr(), grad_weight_.data_ptr(),
                          num_tokens, vocab_size_, embed_dim_, vocab_size_, "default");
  }

  void collect_parameters(std::vector<Tensor<T> *> &params) override { params.push_back(&weight_); }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    grads.push_back(&grad_weight_);
  }

  void clear_gradients() override {
    if (this->initialized_) {
      grad_weight_.fill(T(0));
    }
  }

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    std::vector<size_t> out = input_shape;
    if (out.size() >= 2)
      out[1] = embed_dim_;
    else {
      throw std::runtime_error("EmbeddingLayer::compute_output_shape: Input tensor must have "
                               "shape [N, 1, H, W] or [N, L] where L is the number of tokens.");
    }
    return out;
  }

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override {
    return 0; // Memory bound
  }

  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override { return 0; }

  std::string type() const override { return "embedding"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["vocab_size"] = vocab_size_;
    config.parameters["embed_dim"] = embed_dim_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<EmbeddingLayer<T>>(vocab_size_, embed_dim_, this->name_);
  }

  size_t cached_memory_bytes() const override {
    size_t size = 0;
    for (const auto &kv : micro_batch_inputs_) {
      size += kv.second.size() * sizeof(T);
    }
    return size;
  }
};

} // namespace tnn
