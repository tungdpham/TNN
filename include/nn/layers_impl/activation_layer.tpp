/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/activation_layer.hpp"
#include "ops/ops.hpp"
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
const Tensor<T> &ActivationLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  // const Tensor<T> *current =
  //     input.device() == this->device_ ? &input : input.to_device(this->device_);
  const Tensor<T> *current = &input;
  micro_batch_inputs_[micro_batch_id] = current->clone();
  Tensor<T> &output = this->get_output_buffer(micro_batch_id, current->shape());
  ops::copy(current->data_ptr(), output.data_ptr(), current->size());
  activation_->apply(output);
  return output;
}

template <typename T>
const Tensor<T> &ActivationLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  // const Tensor<T> &current_gradient =
  //     gradient.device() == this->device_ ? gradient : gradient.to_device(this->device_);
  const Tensor<T> *current_gradient = &gradient;
  auto it = micro_batch_inputs_.find(micro_batch_id);
  assert(it != micro_batch_inputs_.end() && "No stored input for given micro_batch_id");
  const Tensor<T> &last_input = it->second;
  Tensor<T> &grad = this->get_gradient_buffer(micro_batch_id, last_input.shape());
  ops::copy(current_gradient->data_ptr(), grad.data_ptr(), current_gradient->size());
  activation_->compute_gradient_inplace(last_input, grad);
  return grad;
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
