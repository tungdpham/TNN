/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/parameterized_layer.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace tnn {

template <typename T = float> class PositionalEmbeddingLayer : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  size_t seq_len_;
  Tensor<T> pos_embedding_;
  Tensor<T> pos_embedding_gradients_;

public:
  PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                           const std::string &name = "pos_embedding")
      : ParameterizedLayer<T>(name), embed_dim_(embed_dim), seq_len_(seq_len) {}

  void initialize_params() override {
    pos_embedding_ = Tensor<T>({1, embed_dim_, seq_len_, 1}, this->device_);
    pos_embedding_gradients_ = Tensor<T>({1, embed_dim_, seq_len_, 1}, this->device_);

    T fan_in = static_cast<T>(embed_dim_);
    T bound = static_cast<T>(1.0) / std::sqrt(fan_in);

    if (this->use_seed_) {
      pos_embedding_.fill_random_uniform(-bound, bound, this->srand_seed_);
    } else {
      pos_embedding_.fill_random_uniform(-bound, bound);
    }
    pos_embedding_gradients_.fill(T(0));
  }

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    if (!this->initialized_) {
      throw std::runtime_error(
          "Layer parameters not initialized. Call initialize() before forward.");
    }

    if (input.dims() != 4) {
      throw std::runtime_error(
          "PositionalEmbeddingLayer: Input tensor must be 4-dimensional (NCHW)");
    }

    size_t batch_size = input.dimension(0);
    size_t channels = input.dimension(1);
    size_t height = input.dimension(2);
    size_t width = input.dimension(3);
    size_t length = input.stride(1);

    if (channels != embed_dim_) {
      throw std::runtime_error("PositionalEmbeddingLayer: Input channels must match embed_dim");
    }
    if (length != seq_len_) {
      if (height != seq_len_ || width != 1) {
        throw std::runtime_error("PositionalEmbeddingLayer: Input sequence length mismatch");
      }
    }

    output.ensure(input.shape(), this->device_);

    auto &out_ptr = output.data_ptr();
    auto &in_ptr = input.data_ptr();
    auto &pos_ptr = pos_embedding_.data_ptr();

    size_t size = channels * seq_len_;

    for (size_t i = 0; i < batch_size; ++i) {
      // output[i] = input[i] + pos_embedding
      if (this->device_->device_type() == DeviceType::CPU) {
        create_cpu_task("default", ops::cpu::add<T>, in_ptr.get() + i * size, pos_ptr.get(),
                        out_ptr.get() + i * size, size);
      }
#ifdef USE_CUDA
      else if (this->device_->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::cuda_add<T>, in_ptr.get() + i * size, pos_ptr.get(),
                        out_ptr.get() + i * size, size);
      }
#endif
    }
  }

  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override {
    if (gradient.dims() != 4) {
      throw std::runtime_error(
          "PositionalEmbeddingLayer: Gradient tensor must be 4-dimensional (NCHW)");
    }

    size_t batch_size = gradient.dimension(0);
    size_t channels = gradient.dimension(1);
    size_t height = gradient.dimension(2);
    size_t width = gradient.dimension(3);

    size_t size = channels * height * width;
    grad_input.ensure(gradient.shape(), this->device_);

    // grad_input = gradient
    ops::copy(gradient.data_ptr(), grad_input.data_ptr(), gradient.size());

    const auto &grad_ptr = gradient.data_ptr();
    auto &pos_grad_ptr = pos_embedding_gradients_.data_ptr();

    for (size_t i = 0; i < batch_size; ++i) {
      // Accumulate gradient to pos_embedding_gradients_
      if (this->device_->device_type() == DeviceType::CPU) {
        create_cpu_task("default", ops::cpu::add<T>, pos_grad_ptr.get(), grad_ptr.get() + i * size,
                        pos_grad_ptr.get(), size);
      }
#ifdef USE_CUDA
      else if (this->device_->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::cuda_add<T>, pos_grad_ptr.get(), grad_ptr.get() + i * size,
                        pos_grad_ptr.get(), size);
      }
#endif
    }
  }

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override { return 0; }

  std::string type() const override { return "pos_embedding"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["embed_dim"] = embed_dim_;
    config.parameters["seq_len"] = seq_len_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<PositionalEmbeddingLayer<T>>(embed_dim_, seq_len_, this->name_);
  }

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape;
  }

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override {
    params.push_back(&pos_embedding_);
  }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    grads.push_back(&pos_embedding_gradients_);
  }
};

} // namespace tnn
