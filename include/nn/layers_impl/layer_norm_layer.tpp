/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/cpu/layer_norm_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/layer_norm_ops.hpp"
#endif
#include "nn/layers_impl/layer_norm_layer.hpp"
#include "ops/ops.hpp"
#include <stdexcept>

namespace tnn {

template <typename T>
LayerNormLayer<T>::LayerNormLayer(size_t normalized_shape, T epsilon, bool affine,
                                  const std::string &name)
    : ParameterizedLayer<T>(name), normalized_shape_(normalized_shape), epsilon_(epsilon),
      affine_(affine) {}

template <typename T> void LayerNormLayer<T>::init_params() {
  if (this->initialized_)
    return;

  if (affine_) {
    gamma_ = Tensor<T>({normalized_shape_}, this->device_);
    beta_ = Tensor<T>({normalized_shape_}, this->device_);
    gamma_.fill(T(1));
    beta_.fill(T(0));

    gamma_gradients_ = Tensor<T>({normalized_shape_}, this->device_);
    beta_gradients_ = Tensor<T>({normalized_shape_}, this->device_);
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));
  }

  this->initialized_ = true;
}

template <typename T>
void LayerNormLayer<T>::forward_impl(const Tensor<T> &input, Tensor<T> &output,
                                     size_t micro_batch_id) {
  if (this->is_training_) {
    Tensor<T> &cached_input = micro_batch_inputs_[micro_batch_id];
    cached_input.ensure(input.shape(), this->device_);
    ops::copy(input.data_ptr(), cached_input.data_ptr(), input.size());
  }
  const auto &shape = input.shape();
  size_t last_dim = shape.back();

  if (last_dim != normalized_shape_) {
    throw std::invalid_argument("Input last dimension (" + std::to_string(last_dim) +
                                ") must match normalized_shape (" +
                                std::to_string(normalized_shape_) + ") in LayerNormLayer");
  }

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  output.ensure(shape, this->device_);

  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  const T *gamma_ptr = affine_ ? gamma_.data() : nullptr;
  const T *beta_ptr = affine_ ? beta_.data() : nullptr;

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::layer_norm::layer_norm_forward<T>, current->data(),
                    output.data(), gamma_ptr, beta_ptr, batch_size, channels, epsilon_);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::layer_norm::layer_norm_forward<T>, current->data(),
                    output.data(), gamma_ptr, beta_ptr, batch_size, channels, epsilon_);
#else
    throw std::runtime_error("CUDA support for LayerNormLayer is not yet implemented.");
#endif
  }
}

template <typename T>
void LayerNormLayer<T>::backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                      size_t micro_batch_id) {
  if (micro_batch_inputs_.find(micro_batch_id) == micro_batch_inputs_.end()) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const Tensor<T> &input = micro_batch_inputs_[micro_batch_id];

  const auto &shape = input.shape();
  grad_input.ensure(shape, this->device_);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  const T *gamma_ptr = affine_ ? gamma_.data() : nullptr;
  T *gamma_grad_ptr = affine_ ? gamma_gradients_.data() : nullptr;
  T *beta_grad_ptr = affine_ ? beta_gradients_.data() : nullptr;

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::layer_norm::layer_norm_backward<T>, gradient.data(),
                    input.data(), gamma_ptr, grad_input.data(), gamma_grad_ptr, beta_grad_ptr,
                    batch_size, channels, epsilon_);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::layer_norm::layer_norm_backward<T>, gradient.data(),
                    input.data(), gamma_ptr, grad_input.data(), gamma_grad_ptr, beta_grad_ptr,
                    batch_size, channels, epsilon_);
#else
    throw std::runtime_error("CUDA support for LayerNormLayer is not yet implemented.");
#endif
  }
}

template <typename T> void LayerNormLayer<T>::clear_gradients() {
  if (affine_) {
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));
  }
}

template <typename T>
uint64_t LayerNormLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2)
    return 0;
  size_t elements = 1;
  for (size_t s : input_shape)
    elements *= s;
  return elements * 8;
}

template <typename T>
uint64_t LayerNormLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2)
    return 0;
  size_t elements = 1;
  for (size_t s : input_shape)
    elements *= s;
  return elements * 16;
}

template <typename T> LayerConfig LayerNormLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["normalized_shape"] = normalized_shape_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["affine"] = affine_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> LayerNormLayer<T>::clone() const {
  return std::make_unique<LayerNormLayer<T>>(normalized_shape_, epsilon_, affine_, this->name_);
}

template <typename T> void LayerNormLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  if (affine_) {
    params.push_back(&gamma_);
    params.push_back(&beta_);
  }
}

template <typename T> void LayerNormLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  if (affine_) {
    grads.push_back(&gamma_gradients_);
    grads.push_back(&beta_gradients_);
  }
}

} // namespace tnn
