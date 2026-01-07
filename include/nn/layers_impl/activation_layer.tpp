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
ActivationLayer<T>::ActivationLayer(std::unique_ptr<EWActivationFunction<T>> activation,
                                    const std::string &name)
    : StatelessLayer<T>(name), activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Function function cannot be null");
  }
}

template <typename T>
void ActivationLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  if (it_input != micro_batch_inputs_.end()) {
    it_input->second.ensure(current->shape(), this->device_);
    ops::copy(current->data_ptr(), it_input->second.data_ptr(), current->size());
  } else {
    micro_batch_inputs_[micro_batch_id] = current->clone();
  }

  output.ensure(current->shape(), this->device_);
  activation_->apply(*current, output);
}

template <typename T>
void ActivationLayer<T>::backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                  size_t micro_batch_id) {
  const Tensor<T> *current_gradient = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_gradient = &device_gradient;
  }

  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  if (it_input == micro_batch_inputs_.end()) {
    throw std::runtime_error("Input for micro batch not found");
  }
  const Tensor<T> &input = it_input->second;
  grad_input.ensure(input.shape(), this->device_);

  activation_->compute_gradient(input, *current_gradient, grad_input);
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
