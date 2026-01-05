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

template <typename T> void LayerNormLayer<T>::initialize_params() {
  if (this->initialized_)
    return;

  if (affine_) {
    gamma_ = Tensor<T>({normalized_shape_, 1, 1, 1}, this->device_);
    beta_ = Tensor<T>({normalized_shape_, 1, 1, 1}, this->device_);
    gamma_.fill(T(1));
    beta_.fill(T(0));

    gamma_gradients_ = Tensor<T>({normalized_shape_, 1, 1, 1}, this->device_);
    beta_gradients_ = Tensor<T>({normalized_shape_, 1, 1, 1}, this->device_);
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));
  }

  this->initialized_ = true;
}

template <typename T>
void LayerNormLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {
  // Cache input
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  if (it_input == micro_batch_inputs_.end()) {
    micro_batch_inputs_[micro_batch_id] = input.clone();
  } else {
    micro_batch_inputs_[micro_batch_id].ensure(input.shape());
    ops::copy(input.data_ptr(), micro_batch_inputs_[micro_batch_id].data_ptr(), input.size());
  }

  if (input.channels() != normalized_shape_) {
    throw std::invalid_argument("Input channels must match normalized_shape in LayerNormLayer");
  }

  output.ensure(input.shape(), this->device_);

  size_t batch_size = input.batch_size();
  size_t channels = input.channels();
  size_t height = input.height();
  size_t width = input.width();
  size_t spatial_size = height * width;

  const T *gamma_ptr = affine_ ? gamma_.data() : nullptr;
  const T *beta_ptr = affine_ ? beta_.data() : nullptr;

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::layer_norm::layer_norm_forward<T>,
                    micro_batch_inputs_[micro_batch_id].data(), output.data(), gamma_ptr, beta_ptr,
                    batch_size, channels, spatial_size, epsilon_);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::layer_norm::layer_norm_forward<T>,
                    micro_batch_inputs_[micro_batch_id].data(), output.data(), gamma_ptr, beta_ptr,
                    batch_size, channels, spatial_size, epsilon_);
#else
    throw std::runtime_error("CUDA support for LayerNormLayer is not yet implemented.");
#endif
  }
}

template <typename T>
void LayerNormLayer<T>::backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                 size_t micro_batch_id) {
  if (micro_batch_inputs_.find(micro_batch_id) == micro_batch_inputs_.end()) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const Tensor<T> &input = micro_batch_inputs_[micro_batch_id];

  grad_input.ensure(input.shape(), this->device_);

  size_t batch_size = input.batch_size();
  size_t channels = input.channels();
  size_t height = input.height();
  size_t width = input.width();
  size_t spatial_size = height * width;

  const T *gamma_ptr = affine_ ? gamma_.data() : nullptr;
  T *gamma_grad_ptr = affine_ ? gamma_gradients_.data() : nullptr;
  T *beta_grad_ptr = affine_ ? beta_gradients_.data() : nullptr;

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::layer_norm::layer_norm_backward<T>, gradient.data(),
                    input.data(), gamma_ptr, grad_input.data(), gamma_grad_ptr, beta_grad_ptr,
                    batch_size, channels, spatial_size, epsilon_);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::layer_norm::layer_norm_backward<T>, gradient.data(),
                    input.data(), gamma_ptr, grad_input.data(), gamma_grad_ptr, beta_grad_ptr,
                    batch_size, channels, spatial_size, epsilon_);
#else
    throw std::runtime_error("CUDA support for LayerNormLayer is not yet implemented.");
#endif
  }
}

template <typename T>
uint64_t LayerNormLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2)
    return 0;
  size_t elements = 1;
  for (size_t s : input_shape)
    elements *= s;
  // Mean: N adds + 1 div. Var: N subs + N muls + N adds + 1 div. Norm: N subs + N muls. Scale: N
  // muls + N adds. approx 8 ops per element
  return elements * 8;
}

template <typename T>
uint64_t LayerNormLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2)
    return 0;
  size_t elements = 1;
  for (size_t s : input_shape)
    elements *= s;
  return elements * 16;
}

template <typename T>
uint64_t LayerNormLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  return forward_complexity(input_shape);
}

template <typename T>
uint64_t LayerNormLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  return backward_complexity(input_shape);
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
