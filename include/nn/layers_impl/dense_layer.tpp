/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/dense_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cpu/dense_ops.hpp"
#include "cuda/dense_ops.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"

namespace tnn {

template <typename T>
DenseLayer<T>::DenseLayer(size_t input_features, size_t output_features,
                          std::unique_ptr<ActivationFunction<T>> activation, bool use_bias,
                          const std::string &name)
    : ParameterizedLayer<T>(name), input_features_(input_features),
      output_features_(output_features), use_bias_(use_bias), activation_(std::move(activation)) {}

template <typename T> void DenseLayer<T>::initialize_params() {
  weights_ = Tensor<T>({output_features_, input_features_, 1, 1}, this->device_);
  weight_gradients_ = Tensor<T>({output_features_, input_features_, 1, 1}, this->device_);
  if (use_bias_) {
    bias_ = Tensor<T>({output_features_, 1, 1, 1}, this->device_);
    bias_gradients_ = Tensor<T>({output_features_, 1, 1, 1}, this->device_);
  }
  T fan_in = static_cast<T>(input_features_);
  T fan_out = static_cast<T>(output_features_);
  T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
  weights_.fill_random_normal(T(0), std_dev);
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (!this->initialized_) {
    throw std::runtime_error("Layer parameters not initialized. Call initialize() before forward.");
  }
  micro_batch_inputs_[micro_batch_id] = input.clone();

  const size_t batch_size = input.batch_size();
  const size_t total_input_features = input.channels() * input.height() * input.width();

  if (total_input_features != input_features_) {
    std::cerr << "Input shape: " << total_input_features
              << " features, expected: " << input_features_ << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in DenseLayer");
  }

  const Tensor<T> &current =
      input.device() == this->device_ ? input : input.to_device(this->device_);

  Tensor<T> output({batch_size, output_features_, size_t(1), size_t(1)}, this->device_);

  compute_dense_forward(current.data_ptr(), weights_.data_ptr(), output.data_ptr(), batch_size,
                        input_features_, output_features_);

  if (use_bias_) {
    add_bias_vector(output.data_ptr(), bias_.data_ptr(), batch_size, output_features_);
  }

  micro_batch_pre_activations_[micro_batch_id] = output.clone();

  if (activation_) {
    activation_->apply(output);
  }

  return output;
}

template <typename T>
Tensor<T> DenseLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  if (!this->initialized_) {
    throw std::runtime_error(
        "Layer parameters not initialized. Call initialize() before backward.");
  }
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end()) {
    for (const auto &pair : micro_batch_inputs_) {
      std::cout << "Cached micro-batch IDs: " << pair.first << std::endl;
    }
    throw std::runtime_error("No cached input found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }
  if (it_pre_act == micro_batch_pre_activations_.end()) {
    throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  if (gradient.shape() != it_pre_act->second.shape()) {
    throw std::invalid_argument("Gradient output shape does not match cached pre-activation shape");
  }

  const Tensor<T> &last_input = it_input->second;
  size_t batch_size = last_input.batch_size();
  Tensor<T> grad_input(last_input.shape(), this->device_);

  Tensor<T> current_grad =
      gradient.device() == this->device_ ? gradient.clone() : gradient.to_device(this->device_);

  if (activation_) {
    activation_->compute_gradient_inplace(it_pre_act->second, current_grad);
  }

  compute_weight_gradients(last_input.data_ptr(), current_grad.data_ptr(),
                           weight_gradients_.data_ptr(), batch_size, input_features_,
                           output_features_);

  if (use_bias_) {
    compute_bias_gradients(current_grad.data_ptr(), bias_gradients_.data_ptr(), batch_size,
                           output_features_);
  }

  compute_input_gradients(current_grad.data_ptr(), weights_.data_ptr(), grad_input.data_ptr(),
                          batch_size, input_features_, output_features_);

  return grad_input;
}

template <typename T>
void DenseLayer<T>::compute_dense_forward(const device_ptr<T[]> &input_data,
                                          const device_ptr<T[]> &weight_data,
                                          device_ptr<T[]> &output_data, const size_t batch_size,
                                          const size_t input_features,
                                          const size_t output_features) const {
  if (input_data.getDeviceType() != weight_data.getDeviceType() ||
      input_data.getDeviceType() != output_data.getDeviceType()) {
    throw std::runtime_error(
        "All device pointers must be on the same device type for compute_dense_forward.");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    cpu::dense::compute_dense_forward(input_data.get(), weight_data.get(), output_data.get(),
                                      batch_size, input_features, output_features);
  }
#ifdef USE_CUDA
  else if (input_data.getDeviceType() == DeviceType::GPU) {
    cuda::dense::compute_dense_forward(input_data.get(), weight_data.get(), output_data.get(),
                                       batch_size, input_features, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_dense_forward.");
  }
}

template <typename T>
void DenseLayer<T>::compute_weight_gradients(const device_ptr<T[]> &input_data,
                                             const device_ptr<T[]> &gradient_data,
                                             device_ptr<T[]> &weight_grad_data,
                                             const size_t batch_size, const size_t input_features,
                                             const size_t output_features) const {
  if (input_data.getDeviceType() != gradient_data.getDeviceType() ||
      input_data.getDeviceType() != weight_grad_data.getDeviceType()) {
    throw std::runtime_error(
        "All device pointers must be on the same device type for compute_weight_gradients.");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    cpu::dense::compute_weight_gradients(input_data.get(), gradient_data.get(),
                                         weight_grad_data.get(), batch_size, input_features,
                                         output_features);
  }
#ifdef USE_CUDA
  else if (input_data.getDeviceType() == DeviceType::GPU) {
    cuda::dense::compute_weight_gradients(input_data.get(), gradient_data.get(),
                                          weight_grad_data.get(), batch_size, input_features,
                                          output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_weight_gradients.");
  }
}

template <typename T>
void DenseLayer<T>::compute_input_gradients(const device_ptr<T[]> &gradient_data,
                                            const device_ptr<T[]> &weight_data,
                                            device_ptr<T[]> &grad_input_data, size_t batch_size,
                                            size_t input_features, size_t output_features) const {
  if (gradient_data.getDeviceType() != weight_data.getDeviceType() ||
      gradient_data.getDeviceType() != grad_input_data.getDeviceType()) {
    throw std::runtime_error(
        "All device pointers must be on the same device type for compute_input_gradients.");
  }
  if (gradient_data.getDeviceType() == DeviceType::CPU) {
    cpu::dense::compute_input_gradients(gradient_data.get(), weight_data.get(),
                                        grad_input_data.get(), batch_size, input_features,
                                        output_features);
  }
#ifdef USE_CUDA
  else if (gradient_data.getDeviceType() == DeviceType::GPU) {
    cuda::dense::compute_input_gradients(gradient_data.get(), weight_data.get(),
                                         grad_input_data.get(), batch_size, input_features,
                                         output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_input_gradients.");
  }
}

template <typename T>
void DenseLayer<T>::compute_bias_gradients(const device_ptr<T[]> &current_grad_data,
                                           const device_ptr<T[]> &bias_gradient_data,
                                           size_t batch_size, size_t output_features) const {
  if (current_grad_data.getDeviceType() != bias_gradient_data.getDeviceType()) {
    throw std::runtime_error("Device type mismatch in compute_bias_gradients");
  }
  if (current_grad_data.getDeviceType() == DeviceType::CPU) {
    cpu::dense::compute_bias_gradients<T>(current_grad_data.get(), bias_gradient_data.get(),
                                          batch_size, output_features);
  }
#ifdef USE_CUDA
  else {
    cuda::dense::compute_bias_gradients<T>(current_grad_data.get(), bias_gradient_data.get(),
                                           batch_size, output_features);
  }
#endif
}

template <typename T>
void DenseLayer<T>::add_bias_vector(device_ptr<T[]> &output_data, const device_ptr<T[]> &bias_data,
                                    size_t batch_size, size_t output_features) const {
  if (output_data.getDeviceType() != bias_data.getDeviceType()) {
    throw std::runtime_error("Device type mismatch in add_bias_vector");
  }
  if (output_data.getDeviceType() == DeviceType::CPU) {
    cpu::dense::add_bias_vector<T>(output_data.get(), bias_data.get(), batch_size, output_features);
  }
#ifdef USE_CUDA
  else {
    cuda::dense::add_bias_vector<T>(output_data.get(), bias_data.get(), batch_size,
                                    output_features);
  }
#endif
}

template <typename T> std::string DenseLayer<T>::type() const { return "dense"; }

template <typename T> LayerConfig DenseLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["input_features"] = input_features_;
  config.parameters["output_features"] = output_features_;
  config.parameters["use_bias"] = use_bias_;
  config.parameters["activation"] = activation_ ? activation_->name() : std::string("none");
  config.parameters["optimized"] = std::string("native");
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> DenseLayer<T>::clone() const {
  auto activation_clone = activation_ ? activation_->clone() : nullptr;
  return std::make_unique<DenseLayer<T>>(input_features_, output_features_,
                                         std::move(activation_clone), use_bias_, this->name_);
}

template <typename T>
std::vector<size_t>
DenseLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("DenseLayer expects 4D input");
  }
  return {input_shape[0], output_features_, 1, 1};
}

template <typename T> void DenseLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T> void DenseLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T> void DenseLayer<T>::clear_gradients() {
  weight_gradients_.fill(T(0));
  if (use_bias_) {
    bias_gradients_.fill(T(0));
  }
}

template <typename T>
std::unique_ptr<Layer<T>> DenseLayer<T>::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");
  std::string activation_name = config.get<std::string>("activation");

  std::unique_ptr<ActivationFunction<T>> activation;
  if (activation_name != "none") {

    ActivationFactory<T>::register_defaults();
    activation = ActivationFactory<T>::create(activation_name);
  }

  return std::make_unique<DenseLayer<T>>(input_features, output_features, std::move(activation),
                                         use_bias, config.name);
}

template <typename T>
uint64_t DenseLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];

  uint64_t gemm_flops = 2ULL * batch_size * input_features_ * output_features_;

  uint64_t bias_flops = use_bias_ ? (batch_size * output_features_) : 0;

  uint64_t activation_flops = activation_ ? (batch_size * output_features_) : 0;

  return gemm_flops + bias_flops + activation_flops;
}

template <typename T>
uint64_t DenseLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];

  uint64_t activation_grad_flops = activation_ ? (batch_size * output_features_) : 0;

  uint64_t weight_grad_flops = 2ULL * input_features_ * batch_size * output_features_;

  uint64_t bias_grad_flops = use_bias_ ? (batch_size * output_features_) : 0;

  uint64_t input_grad_flops = 2ULL * batch_size * output_features_ * input_features_;

  return activation_grad_flops + weight_grad_flops + bias_grad_flops + input_grad_flops;
}

template <typename T>
uint64_t DenseLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {

  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t DenseLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {

  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class DenseLayer<float>;
// template class DenseLayer<double>;

} // namespace tnn
