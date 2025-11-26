/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/dropout_layer.hpp"

#include <memory>
#include <stdexcept>

#include "cpu/dropout_ops.hpp"
#include "device/task.hpp"
#ifdef USE_CUDA
#include "cuda/dropout_ops.hpp"
#endif

namespace tnn {

template <typename T>
DropoutLayer<T>::DropoutLayer(T dropout_rate, const std::string &name)
    : StatelessLayer<T>(name), dropout_rate_(dropout_rate), generator_(std::random_device{}()) {
  if (dropout_rate < T(0) || dropout_rate >= T(1)) {
    throw std::invalid_argument("Dropout rate must be in [0, 1)");
  }
}

template <typename T>
const Tensor<T> &DropoutLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (!this->is_training_) {
    return input;
  }

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  Tensor<T> mask(current->shape(), this->device_);
  Tensor<T> &output = this->get_buffer(current->shape());

  auto forward_task = compute_dropout_forward(*current, output, mask);
  auto err = forward_task->sync();
  if (err != ErrorStatus{}) {
    throw std::runtime_error("DropoutLayer forward computation failed");
  }

  micro_batch_masks_[micro_batch_id] = mask.clone();
  return output;
}

template <typename T>
const Tensor<T> &DropoutLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  if (!this->is_training_) {
    return gradient;
  }

  const Tensor<T> *current_gradient = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_gradient = &device_gradient;
  }

  auto it_mask = micro_batch_masks_.find(micro_batch_id);
  if (it_mask == micro_batch_masks_.end()) {
    throw std::runtime_error("No cached mask found for micro-batch ID in DropoutLayer: " +
                             std::to_string(micro_batch_id));
  }
  const Tensor<T> &mask = it_mask->second;

  Tensor<T> &grad_input = this->get_buffer(current_gradient->shape());

  grad_input = (*current_gradient) * mask;

  return grad_input;
}

template <typename T>
std::unique_ptr<Task> DropoutLayer<T>::compute_dropout_forward(const Tensor<T> &input,
                                                               Tensor<T> &output, Tensor<T> &mask) {
  size_t batch_size = input.batch_size();
  size_t channels = input.channels();
  size_t spatial_size = input.height() * input.width();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::compute_dropout_forward<T>, input.data(), output.data(),
                           mask.data(), batch_size, channels, spatial_size, dropout_rate_);
  }
#ifdef USE_CUDA
  else if (input.device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::compute_dropout_forward<T>, input.data(), output.data(),
                           mask.data(), batch_size, channels, spatial_size, dropout_rate_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_dropout_forward");
  }
}

template <typename T> std::string DropoutLayer<T>::type() const { return "dropout"; }

template <typename T> LayerConfig DropoutLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["dropout_rate"] = dropout_rate_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> DropoutLayer<T>::clone() const {
  return std::make_unique<DropoutLayer<T>>(dropout_rate_, this->name_);
}

template <typename T>
std::vector<size_t>
DropoutLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T>
std::unique_ptr<Layer<T>> DropoutLayer<T>::create_from_config(const LayerConfig &config) {
  T dropout_rate = config.get<T>("dropout_rate");
  return std::make_unique<DropoutLayer<T>>(dropout_rate, config.name);
}

template <typename T>
uint64_t DropoutLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  if (!this->is_training_) {
    return 0;
  }

  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  uint64_t rng_flops = num_elements;
  uint64_t mask_flops = num_elements;
  uint64_t scale_flops = static_cast<uint64_t>((1.0 - dropout_rate_) * num_elements);

  return rng_flops + mask_flops + scale_flops;
}

template <typename T>
uint64_t DropoutLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  if (!this->is_training_) {
    return 0;
  }

  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return num_elements;
}

template <typename T>
uint64_t DropoutLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {

  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t DropoutLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class DropoutLayer<float>;

} // namespace tnn
