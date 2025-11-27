/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/flatten_layer.hpp"
#include "ops/ops.hpp"
#include <stdexcept>

namespace tnn {

template <typename T>
FlattenLayer<T>::FlattenLayer(const std::string &name) : StatelessLayer<T>(name) {}

template <typename T>
void FlattenLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {
  micro_batch_original_shapes_[micro_batch_id] = input.shape();

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  size_t batch_size = current->batch_size();
  size_t features = current->channels() * current->height() * current->width();

  output.ensure(std::vector<size_t>{batch_size, features, 1, 1});

  ops::copy(current->data_ptr(), output.data_ptr(), current->size());
}

template <typename T>
void FlattenLayer<T>::backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                               size_t micro_batch_id) {
  auto it = micro_batch_original_shapes_.find(micro_batch_id);
  if (it == micro_batch_original_shapes_.end()) {
    throw std::runtime_error("No cached shape found for micro-batch ID in FlattenLayer: " +
                             std::to_string(micro_batch_id));
  }
  const std::vector<size_t> &original_shape = it->second;

  const Tensor<T> *current_grad = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_grad = &device_gradient;
  }

  grad_input.ensure(original_shape);
  ops::copy(current_grad->data_ptr(), grad_input.data_ptr(), current_grad->size());
}

template <typename T> std::string FlattenLayer<T>::type() const { return "flatten"; }

template <typename T> LayerConfig FlattenLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> FlattenLayer<T>::clone() const {
  return std::make_unique<FlattenLayer<T>>(this->name_);
}

template <typename T>
std::vector<size_t>
FlattenLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("FlattenLayer expects 4D input");
  }

  size_t features = input_shape[1] * input_shape[2] * input_shape[3];
  return {input_shape[0], features, 1, 1};
}

template <typename T>
std::unique_ptr<Layer<T>> FlattenLayer<T>::create_from_config(const LayerConfig &config) {
  return std::make_unique<FlattenLayer<T>>(config.name);
}

template <typename T>
uint64_t FlattenLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template <typename T>
uint64_t FlattenLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {

  return 0;
}

template <typename T>
uint64_t FlattenLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {

  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t FlattenLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {

  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class FlattenLayer<float>;
template class FlattenLayer<double>;

} // namespace tnn
