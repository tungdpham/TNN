/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/activation_layer.hpp"
namespace tnn {

template <typename T>
ActivationLayer<T>::ActivationLayer(std::unique_ptr<ActivationFunction<T>> activation,
                                    const std::string &name)
    : StatelessLayer<T>(name), activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Activation function cannot be null");
  }
}

template <typename T>
Tensor<T> ActivationLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  micro_batch_inputs_[micro_batch_id] = input.clone();
  Tensor<T> output = input.clone();
  activation_->apply(output);
  return output;
}

template <typename T>
Tensor<T> ActivationLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it = micro_batch_inputs_.find(micro_batch_id);
  assert(it != micro_batch_inputs_.end() && "No stored input for given micro_batch_id");
  const Tensor<T> &last_input = it->second;
  Tensor<T> grad = activation_->compute_gradient(last_input, &gradient);
  return grad;
}

template <typename T>
void ActivationLayer<T>::forward_inplace(Tensor<T> &input, size_t micro_batch_id) {
  micro_batch_inputs_[micro_batch_id] = input.clone();
  activation_->apply(input);
}

template <typename T>
void ActivationLayer<T>::backward_inplace(Tensor<T> &gradient, size_t micro_batch_id) {
  auto it = micro_batch_inputs_.find(micro_batch_id);
  assert(it != micro_batch_inputs_.end() && "No stored input for given micro_batch_id");
  const Tensor<T> &last_input = it->second;
  activation_->compute_gradient_inplace(last_input, gradient);
}

template <typename T> std::string ActivationLayer<T>::type() const { return "activation"; }

template <typename T> LayerConfig ActivationLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["activation"] = activation_->name();
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> ActivationLayer<T>::clone() const {
  return std::make_unique<ActivationLayer<T>>(activation_->clone(), this->name_);
}

template <typename T>
std::vector<size_t>
ActivationLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T>
uint64_t ActivationLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return 2 * num_elements;
}

template <typename T>
uint64_t ActivationLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return 2 * num_elements;
}

template <typename T>
uint64_t ActivationLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t ActivationLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class ActivationLayer<float>;
template class ActivationLayer<double>;

} // namespace tnn
