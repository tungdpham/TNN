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
FlattenLayer<T>::FlattenLayer(int start_dim, const std::string &name)
    : StatelessLayer<T>(name), start_dim_(start_dim) {}

template <typename T>
void FlattenLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {
  micro_batch_original_shapes_[micro_batch_id] = input.shape();

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  std::vector<size_t> output_shape = compute_output_shape(current->shape());
  output.ensure(output_shape);

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
  config.parameters["start_dim"] = start_dim_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> FlattenLayer<T>::clone() const {
  return std::make_unique<FlattenLayer<T>>(this->start_dim_, this->name_);
}

template <typename T>
std::vector<size_t>
FlattenLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::invalid_argument("FlattenLayer expects non-empty input shape");
  }

  std::vector<size_t> output_shape;

  output_shape.push_back(input_shape[0]);

  size_t flat_dim = 1;
  int start = std::max(1, start_dim_);

  for (int i = 1; i < start && i < static_cast<int>(input_shape.size()); ++i) {
    output_shape.push_back(input_shape[i]);
  }

  for (size_t i = static_cast<size_t>(start); i < input_shape.size(); ++i) {
    flat_dim *= input_shape[i];
  }

  output_shape.push_back(flat_dim);

  return output_shape;
}

template <typename T>
std::unique_ptr<Layer<T>> FlattenLayer<T>::create_from_config(const LayerConfig &config) {
  int start_dim = config.get<int>("start_dim", 1);
  return std::make_unique<FlattenLayer<T>>(start_dim, config.name);
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
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT64_MAX)));
}

template <typename T>
uint64_t FlattenLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {

  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT64_MAX)));
}

template class FlattenLayer<float>;
template class FlattenLayer<double>;

} // namespace tnn
