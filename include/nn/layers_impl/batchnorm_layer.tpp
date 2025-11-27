/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/task.hpp"
#include "nn/layers_impl/base_layer.hpp"
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

  // Allocate gradients unconditionally to support non-affine mode without invalid pointers
  gamma_gradients_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  beta_gradients_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  gamma_gradients_.fill(T(0));
  beta_gradients_.fill(T(0));

  if (affine_) {
    gamma_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
    beta_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
    gamma_.fill(T(1));
    beta_.fill(T(0));
  }

  running_mean_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  running_var_ = Tensor<T>({num_features_, 1, 1, 1}, this->device_);
  running_mean_.fill(T(0));
  running_var_.fill(T(1));

  this->initialized_ = true;
}

template <typename T>
void BatchNormLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {
  if (input.channels() != num_features_) {
    throw std::invalid_argument("Input channels must match num_features in BatchNormLayer");
  }

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  size_t batch_size, channels, height, width, spatial_size;
  extract_tensor_dimensions(*current, batch_size, channels, height, width, spatial_size);
  if (num_features_ != channels) {
    throw std::invalid_argument("Input channels must match num_features in BatchNormLayer");
  }

  output.ensure(current->shape(), this->device_);

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
    forward_task_ = run_forward_fused(
        current->data_ptr(), batch_mean_fixed_[micro_batch_id],
        micro_batch_inv_std_[micro_batch_id], running_mean_.data_ptr(), running_var_.data_ptr(),
        gamma_.data_ptr(), beta_.data_ptr(), output.data_ptr(),
        micro_batch_normalized_[micro_batch_id], batch_size, channels, spatial_size, "default");
  } else {
    forward_task_ =
        compute_inference_output(*current, output, batch_size, channels, spatial_size, "default");
  }
}

template <typename T>
void BatchNormLayer<T>::backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                 size_t micro_batch_id) {
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);

  if (it_normalized == micro_batch_normalized_.end()) {
    throw std::runtime_error("No cached data found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  auto it_inv_std = micro_batch_inv_std_.find(micro_batch_id);
  if (it_inv_std == micro_batch_inv_std_.end()) {
    throw std::runtime_error("No cached inv_std found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor<T> *current_gradient = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_gradient = &device_gradient;
  }

  const size_t batch_size = current_gradient->batch_size();
  const size_t channels = current_gradient->channels();
  const size_t height = current_gradient->height();
  const size_t width = current_gradient->width();
  const size_t spatial_size = height * width;

  grad_input.ensure(current_gradient->shape(), this->device_);

  auto bwd_task =
      run_backward_fused(current_gradient->data_ptr(), it_normalized->second, it_inv_std->second,
                         gamma_.data_ptr(), gamma_gradients_.data_ptr(), beta_gradients_.data_ptr(),
                         grad_input.data_ptr(), batch_size, channels, spatial_size, "default");
}

template <typename T>
std::unique_ptr<Task> BatchNormLayer<T>::run_forward_fused(
    const device_ptr<T[]> &input, device_ptr<T[]> &batch_mean_fixed, device_ptr<T[]> &batch_inv_std,
    device_ptr<T[]> &running_mean, device_ptr<T[]> &running_var, const device_ptr<T[]> &gamma,
    const device_ptr<T[]> &beta, device_ptr<T[]> &output, device_ptr<T[]> &norm_cache,
    size_t batch_size, size_t channels, size_t spatial_size, const std::string &flow_id) {
#ifdef USE_CUDA
  if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::batchnorm::run_forward_fused<T>, input.get(),
                           batch_mean_fixed.get(), batch_inv_std.get(), running_mean.get(),
                           running_var.get(), affine_ ? gamma.get() : nullptr,
                           affine_ ? beta.get() : nullptr, output.get(), norm_cache.get(),
                           batch_size, channels, spatial_size, momentum_, epsilon_, affine_);
  } else
#endif
  {
    return create_cpu_task("default", cpu::batchnorm::run_forward_fused<T>, input.get(),
                           batch_mean_fixed.get(), batch_inv_std.get(), running_mean.get(),
                           running_var.get(), affine_ ? gamma.get() : nullptr,
                           affine_ ? beta.get() : nullptr, output.get(), norm_cache.get(),
                           batch_size, channels, spatial_size, momentum_, epsilon_, affine_);
  }
}

template <typename T>
std::unique_ptr<Task> BatchNormLayer<T>::run_backward_fused(
    const device_ptr<T[]> &grad_output, const device_ptr<T[]> &norm_input,
    const device_ptr<T[]> &inv_std, const device_ptr<T[]> &gamma, device_ptr<T[]> &d_gamma,
    device_ptr<T[]> &d_beta, device_ptr<T[]> &grad_input, size_t batch_size, size_t channels,
    size_t spatial_size, const std::string &flow_id) {
#ifdef USE_CUDA
  if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::batchnorm::run_backward_fused<T>, grad_output.get(),
                           norm_input.get(), inv_std.get(), gamma.get(), d_gamma.get(),
                           d_beta.get(), grad_input.get(), batch_size, channels, spatial_size,
                           affine_);
  } else
#endif
  {
    return create_cpu_task("default", cpu::batchnorm::run_backward_fused<T>, grad_output.get(),
                           norm_input.get(), inv_std.get(), gamma.get(), d_gamma.get(),
                           d_beta.get(), grad_input.get(), batch_size, channels, spatial_size,
                           affine_);
  }
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
}

template <typename T> void BatchNormLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  if (affine_) {
    grads.push_back(&gamma_gradients_);
    grads.push_back(&beta_gradients_);
  }
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

template <typename T> size_t BatchNormLayer<T>::cached_memory_bytes() const {
  size_t total_bytes = 0;

  size_t normalized_cache_size = 0;
  // Cached normalized outputs per micro-batch
  for (const auto &pair : micro_batch_normalized_) {
    normalized_cache_size += pair.second.size() * sizeof(T);
  }
  // std::cout << "Normalized cache size: " << normalized_cache_size << " bytes" << std::endl;

  size_t inv_std_cache_size = 0;
  // Cached inverse stddev per micro-batch
  for (const auto &pair : micro_batch_inv_std_) {
    inv_std_cache_size += pair.second.size() * sizeof(T);
  }
  // std::cout << "Inv std cache size: " << inv_std_cache_size << " bytes" << std::endl;

  size_t mean_cache_size = 0;
  // Cached fixed batch means per micro-batch
  for (const auto &pair : batch_mean_fixed_) {
    mean_cache_size += pair.second.size() * sizeof(T);
  }
  // std::cout << "Mean cache size: " << mean_cache_size << " bytes" << std::endl;

  total_bytes += normalized_cache_size + inv_std_cache_size + mean_cache_size;
  total_bytes += Layer<T>::cached_memory_bytes();
  return total_bytes;
}

template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

} // namespace tnn
