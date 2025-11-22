/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/task.hpp"
#include "nn/layers_impl/batchnorm_layer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "nn/layers_impl/cpu/batchnorm_ops.hpp"
#include "nn/layers_impl/cuda/batchnorm_ops.hpp"

namespace tnn {

template <typename T>
BatchNormLayer<T>::BatchNormLayer(size_t num_features, T epsilon, T momentum, bool affine,
                                  const std::string &name)
    : ParameterizedLayer<T>(name), num_features_(num_features), epsilon_(epsilon),
      momentum_(momentum), affine_(affine) {}

template <typename T> void BatchNormLayer<T>::initialize_params() {
  if (this->initialized_) {
    return;
  }

  if (affine_) {
    gamma_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
    beta_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
    gamma_gradients_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
    beta_gradients_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);

    gamma_.fill(T(1));
    beta_.fill(T(0));
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));
  }

  running_mean_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  running_var_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  running_mean_gradients_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  running_var_gradients_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  running_mean_.fill(T(0));
  running_var_.fill(T(1));
  running_mean_gradients_.fill(T(0));
  running_var_gradients_.fill(T(0));

  this->initialized_ = true;
}

template <typename T>
const Tensor<T> &BatchNormLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != num_features_) {
    throw std::invalid_argument("Input channels must match num_features in BatchNormLayer");
  }

  // const Tensor<T> &current =
  //     input.device() == this->device_ ? input : input.to_device(this->device_);
  const Tensor<T> *current = &input;

  size_t batch_size, channels, height, width, spatial_size;
  extract_tensor_dimensions(*current, batch_size, channels, height, width, spatial_size);

  Tensor<T> &output = this->get_output_buffer(micro_batch_id, current->shape());
  output.fill(T(0));

  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);
  if (it_normalized == micro_batch_normalized_.end()) {
    micro_batch_normalized_[micro_batch_id] = make_array_ptr<T[]>(this->device_, current->size());
  } else {
    micro_batch_normalized_[micro_batch_id].ensure(current->size());
  }

  auto it_batch_mean_fixed = batch_mean_fixed_.find(micro_batch_id);
  if (it_batch_mean_fixed == batch_mean_fixed_.end()) {
    batch_mean_fixed_[micro_batch_id] = make_array_ptr<T[]>(this->device_, num_features_);
  } else {
    batch_mean_fixed_[micro_batch_id].ensure(num_features_);
  }

  auto it_batch_inv_std_fixed = micro_batch_inv_std_.find(micro_batch_id);
  if (it_batch_inv_std_fixed == micro_batch_inv_std_.end()) {
    micro_batch_inv_std_[micro_batch_id] = make_array_ptr<T[]>(this->device_, num_features_);
  } else {
    micro_batch_inv_std_[micro_batch_id].ensure(num_features_);
  }

  if (this->is_training_) {

#ifdef USE_CUDA
    if (this->device_->getDeviceType() == DeviceType::GPU) {
      // Optimized GPU path: Single fused call (2 kernels instead of 4-5)
      auto fwd_task = create_gpu_task(
          "default", cuda::batchnorm::run_forward_fused<T>, input.data_ptr().get(),
          batch_mean_fixed_[micro_batch_id].get(), micro_batch_inv_std_[micro_batch_id].get(),
          running_mean_.data_ptr().get(), running_var_.data_ptr().get(),
          affine_ ? gamma_.data_ptr().get() : nullptr, affine_ ? beta_.data_ptr().get() : nullptr,
          output.data_ptr().get(), micro_batch_normalized_[micro_batch_id].get(), batch_size,
          channels, spatial_size, momentum_, epsilon_, affine_);
    } else
#endif
    {
      // CPU path: Use fused forward matching GPU behavior (stores inv_std instead of std)
      auto fwd_task = create_cpu_task(
          "default", cpu::batchnorm::run_forward_fused<T>, input.data_ptr().get(),
          batch_mean_fixed_[micro_batch_id].get(), micro_batch_inv_std_[micro_batch_id].get(),
          running_mean_.data_ptr().get(), running_var_.data_ptr().get(),
          affine_ ? gamma_.data_ptr().get() : nullptr, affine_ ? beta_.data_ptr().get() : nullptr,
          output.data_ptr().get(), micro_batch_normalized_[micro_batch_id].get(), batch_size,
          channels, spatial_size, momentum_, epsilon_, affine_);
    }
  } else {
    compute_inference_output(input, output, batch_size, channels, spatial_size, "default");
  }

  micro_batch_inputs_[micro_batch_id] = input.clone();
  return output;
}

template <typename T>
const Tensor<T> &BatchNormLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end() || it_normalized == micro_batch_normalized_.end()) {
    throw std::runtime_error("No cached data found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  // const Tensor<T> &current_gradient =
  //     gradient.device() == this->device_ ? gradient : gradient.to_device(this->device_);
  const Tensor<T> *current_gradient = &gradient;

  const Tensor<T> &input = it_input->second;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();
  const size_t spatial_size = height * width;

  Tensor<T> &grad_input = this->get_gradient_buffer(micro_batch_id, input.shape());
  grad_input.fill(T(0));

#ifdef USE_CUDA
  if (this->device_->getDeviceType() == DeviceType::GPU) {
    // Optimized GPU path: Fused backward (2 kernels, no allocations)
    auto it_inv_std = micro_batch_inv_std_.find(micro_batch_id);
    if (it_inv_std == micro_batch_inv_std_.end()) {
      throw std::runtime_error("No cached inv_std found for micro-batch ID: " +
                               std::to_string(micro_batch_id));
    }

    auto bwd_task = create_gpu_task(
        "default", cuda::batchnorm::run_backward_fused<T>, current_gradient->data_ptr().get(),
        micro_batch_normalized_[micro_batch_id].get(), it_inv_std->second.get(),
        affine_ ? gamma_.data_ptr().get() : nullptr,
        affine_ ? gamma_gradients_.data_ptr().get() : nullptr,
        affine_ ? beta_gradients_.data_ptr().get() : nullptr, grad_input.data_ptr().get(),
        batch_size, channels, spatial_size, affine_);
  } else
#endif
  {
    // CPU path: Use fused backward matching GPU behavior (uses inv_std)
    auto it_inv_std = micro_batch_inv_std_.find(micro_batch_id);
    if (it_inv_std == micro_batch_inv_std_.end()) {
      throw std::runtime_error("No cached inv_std found for micro-batch ID: " +
                               std::to_string(micro_batch_id));
    }

    auto bwd_task = create_cpu_task(
        "default", cpu::batchnorm::run_backward_fused<T>, current_gradient->data_ptr().get(),
        micro_batch_normalized_[micro_batch_id].get(), it_inv_std->second.get(),
        affine_ ? gamma_.data_ptr().get() : nullptr,
        affine_ ? gamma_gradients_.data_ptr().get() : nullptr,
        affine_ ? beta_gradients_.data_ptr().get() : nullptr, grad_input.data_ptr().get(),
        batch_size, channels, spatial_size, affine_);
  }

  micro_batch_gradients_[micro_batch_id] = grad_input.clone();
  return grad_input;
}

template <typename T>
std::unique_ptr<Task>
BatchNormLayer<T>::compute_inference_output(const Tensor<T> &input, Tensor<T> &output,
                                            size_t batch_size, size_t channels, size_t spatial_size,
                                            const std::string &flow_id) {
  if (input.device_type() != output.device_type() ||
      input.device_type() != running_mean_.device_type() ||
      running_mean_.device_type() != running_var_.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for inference output");
  }

  if (affine_ && (input.device_type() != gamma_.device_type() ||
                  gamma_.device_type() != beta_.device_type())) {
    throw std::runtime_error("Gamma and beta must be on the same device as input");
  }

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(
        flow_id, cpu::batchnorm::compute_inference_output<T>, input.data_ptr().get(),
        running_mean_.data_ptr().get(), running_var_.data_ptr().get(),
        affine_ ? gamma_.data_ptr().get() : nullptr, affine_ ? beta_.data_ptr().get() : nullptr,
        output.data_ptr().get(), batch_size, channels, spatial_size, epsilon_, affine_);
  }
#ifdef USE_CUDA
  else if (input.device_type() == DeviceType::GPU) {
    return create_gpu_task(
        flow_id, cuda::batchnorm::compute_inference_output<T>, input.data_ptr().get(),
        running_mean_.data_ptr().get(), running_var_.data_ptr().get(),
        affine_ ? gamma_.data_ptr().get() : nullptr, affine_ ? beta_.data_ptr().get() : nullptr,
        output.data_ptr().get(), batch_size, channels, spatial_size, epsilon_, affine_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_inference_output");
  }
  return nullptr;
}

template <typename T>
void BatchNormLayer<T>::extract_tensor_dimensions(const Tensor<T> &input, size_t &batch_size,
                                                  size_t &channels, size_t &height, size_t &width,
                                                  size_t &spatial_size) {
  batch_size = input.batch_size();
  channels = input.channels();
  height = input.height();
  width = input.width();
  spatial_size = height * width;
}

template <typename T> std::string BatchNormLayer<T>::type() const { return "batchnorm"; }

template <typename T> LayerConfig BatchNormLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["num_features"] = num_features_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["momentum"] = momentum_;
  config.parameters["affine"] = affine_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> BatchNormLayer<T>::clone() const {
  return std::make_unique<BatchNormLayer<T>>(num_features_, epsilon_, momentum_, affine_,
                                             this->name_);
}

template <typename T>
std::vector<size_t>
BatchNormLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T> void BatchNormLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  if (affine_) {
    params.push_back(&gamma_);
    params.push_back(&beta_);
  }
  params.push_back(&running_mean_);
  params.push_back(&running_var_);
}

template <typename T> void BatchNormLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  if (affine_) {
    grads.push_back(&gamma_gradients_);
    grads.push_back(&beta_gradients_);
  }
  grads.push_back(&running_mean_gradients_);
  grads.push_back(&running_var_gradients_);
}

template <typename T>
std::unique_ptr<Layer<T>> BatchNormLayer<T>::create_from_config(const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  T epsilon = config.get<T>("epsilon");
  T momentum = config.get<T>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_unique<BatchNormLayer<T>>(num_features, epsilon, momentum, affine, config.name);
}

template <typename T> void BatchNormLayer<T>::clear_gradients() {
  if (affine_) {
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));
  }
}

template <typename T>
uint64_t BatchNormLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  uint64_t mean_flops = batch_size * spatial_size * num_features_;

  uint64_t var_flops = 2 * num_elements + mean_flops;

  uint64_t norm_flops = 3 * num_elements;

  uint64_t affine_flops = affine_ ? (2 * num_elements) : 0;

  return mean_flops + var_flops + norm_flops + affine_flops;
}

template <typename T>
uint64_t BatchNormLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  uint64_t param_grad_flops = affine_ ? (2 * batch_size * spatial_size * num_features_) : 0;

  uint64_t input_grad_flops = 9 * num_elements;

  return param_grad_flops + input_grad_flops;
}

template <typename T>
uint64_t BatchNormLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t BatchNormLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

} // namespace tnn
