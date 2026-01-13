/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/positional_embedding_layer.hpp"
#include "ops/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace tnn {

template <typename T>
PositionalEmbeddingLayer<T>::PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                                                      const std::string &name)
    : ParameterizedLayer<T>(name), embed_dim_(embed_dim), seq_len_(seq_len) {}

template <typename T> void PositionalEmbeddingLayer<T>::init_params() {
  pos_embedding_ = Tensor<T>({seq_len_, embed_dim_}, this->device_);
  pos_embedding_gradients_ = Tensor<T>({seq_len_, embed_dim_}, this->device_);

  T fan_in = static_cast<T>(embed_dim_);
  T bound = static_cast<T>(1.0) / std::sqrt(fan_in);

  if (this->use_seed_) {
    pos_embedding_.fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    pos_embedding_.fill_random_uniform(-bound, bound);
  }
  pos_embedding_gradients_.fill(T(0));
}

template <typename T>
void PositionalEmbeddingLayer<T>::forward_impl(const Tensor<T> &input, Tensor<T> &output,
                                               size_t micro_batch_id) {
  const auto &shape = input.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingLayer: Input tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Input last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Input sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  output.ensure(shape, this->device_);

  auto &out_ptr = output.data_ptr();
  auto &in_ptr = input.data_ptr();
  auto &pos_ptr = pos_embedding_.data_ptr();

  size_t sample_size = seq_len_ * embed_dim_;
  size_t batch_size = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  for (size_t i = 0; i < batch_size; ++i) {
    // output[i] = input[i] + pos_embedding
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", ops::cpu::add<T>, in_ptr.get() + i * sample_size, pos_ptr.get(),
                      out_ptr.get() + i * sample_size, sample_size);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::cuda_add<T>, in_ptr.get() + i * sample_size, pos_ptr.get(),
                      out_ptr.get() + i * sample_size, sample_size);
    }
#endif
  }
}

template <typename T>
void PositionalEmbeddingLayer<T>::backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                                size_t micro_batch_id) {
  const auto &shape = gradient.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingLayer: Gradient tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Gradient last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Gradient sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  size_t sample_size = seq_len_ * embed_dim_;
  size_t batch_size = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  grad_input.ensure(shape, this->device_);

  // grad_input = gradient
  ops::copy(gradient.data_ptr(), grad_input.data_ptr(), gradient.size());

  const auto &grad_ptr = gradient.data_ptr();
  auto &pos_grad_ptr = pos_embedding_gradients_.data_ptr();

  for (size_t i = 0; i < batch_size; ++i) {
    // Accumulate gradient to pos_embedding_gradients_
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", ops::cpu::add<T>, pos_grad_ptr.get(),
                      grad_ptr.get() + i * sample_size, pos_grad_ptr.get(), sample_size);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::cuda_add<T>, pos_grad_ptr.get(),
                      grad_ptr.get() + i * sample_size, pos_grad_ptr.get(), sample_size);
    }
#endif
  }
}

template <typename T>
uint64_t PositionalEmbeddingLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template <typename T>
uint64_t PositionalEmbeddingLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template <typename T> std::string PositionalEmbeddingLayer<T>::type() const {
  return "pos_embedding";
}

template <typename T> LayerConfig PositionalEmbeddingLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["embed_dim"] = embed_dim_;
  config.parameters["seq_len"] = seq_len_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> PositionalEmbeddingLayer<T>::clone() const {
  return std::make_unique<PositionalEmbeddingLayer<T>>(embed_dim_, seq_len_, this->name_);
}

template <typename T>
std::vector<size_t>
PositionalEmbeddingLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T>
void PositionalEmbeddingLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&pos_embedding_);
}

template <typename T>
void PositionalEmbeddingLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&pos_embedding_gradients_);
}

} // namespace tnn
