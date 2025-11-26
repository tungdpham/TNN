/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/task.hpp"
#include "nn/layers_impl/groupnorm_layer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "nn/layers_impl/cpu/groupnorm_ops.hpp"
#include "nn/layers_impl/cuda/groupnorm_ops.hpp"

namespace tnn {

template <typename T>
GroupNormLayer<T>::GroupNormLayer(size_t num_groups, size_t num_channels, T epsilon, bool affine,
                                  const std::string &name)
    : ParameterizedLayer<T>(name), num_groups_(num_groups), num_channels_(num_channels),
      epsilon_(epsilon), affine_(affine) {
  if (num_channels_ % num_groups_ != 0) {
    throw std::invalid_argument("num_channels must be divisible by num_groups in GroupNormLayer");
  }
}

template <typename T> void GroupNormLayer<T>::initialize_params() {
  if (this->initialized_) {
    return;
  }

  // Allocate gradients unconditionally to support non-affine mode without invalid pointers
  gamma_gradients_ = Tensor<T>({num_channels_, 1, 1, 1}, this->device_);
  beta_gradients_ = Tensor<T>({num_channels_, 1, 1, 1}, this->device_);
  gamma_gradients_.fill(T(0));
  beta_gradients_.fill(T(0));

  if (affine_) {
    gamma_ = Tensor<T>({num_channels_, 1, 1, 1}, this->device_);
    beta_ = Tensor<T>({num_channels_, 1, 1, 1}, this->device_);
    gamma_.fill(T(1));
    beta_.fill(T(0));
  }

  this->initialized_ = true;
}

template <typename T>
const Tensor<T> &GroupNormLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != num_channels_) {
    throw std::invalid_argument("Input channels must match num_channels in GroupNormLayer");
  }

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  size_t batch_size, channels, height, width, spatial_size;
  extract_tensor_dimensions(*current, batch_size, channels, height, width, spatial_size);
  if (num_channels_ != channels) {
    throw std::invalid_argument("Input channels must match num_channels in GroupNormLayer");
  }

  Tensor<T> &output = this->get_output_buffer(micro_batch_id, current->shape());

  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);
  if (it_normalized == micro_batch_normalized_.end()) {
    micro_batch_normalized_[micro_batch_id] = make_array_ptr<T[]>(this->device_, current->size());
  } else {
    micro_batch_normalized_[micro_batch_id].ensure(current->size());
  }

  auto it_group_mean = group_mean_.find(micro_batch_id);
  if (it_group_mean == group_mean_.end()) {
    group_mean_[micro_batch_id] = make_array_ptr<T[]>(this->device_, batch_size * num_groups_);
  } else {
    group_mean_[micro_batch_id].ensure(batch_size * num_groups_);
  }

  auto it_group_inv_std = micro_batch_inv_std_.find(micro_batch_id);
  if (it_group_inv_std == micro_batch_inv_std_.end()) {
    micro_batch_inv_std_[micro_batch_id] =
        make_array_ptr<T[]>(this->device_, batch_size * num_groups_);
  } else {
    micro_batch_inv_std_[micro_batch_id].ensure(batch_size * num_groups_);
  }

  std::unique_ptr<Task> fwd_task;
  fwd_task = run_forward_fused(
      current->data_ptr(), group_mean_[micro_batch_id], micro_batch_inv_std_[micro_batch_id],
      gamma_.data_ptr(), beta_.data_ptr(), output.data_ptr(),
      micro_batch_normalized_[micro_batch_id], batch_size, channels, spatial_size, "default");
  if (fwd_task) {
    auto err = fwd_task->sync();
    if (err != ErrorStatus{}) {
      throw std::runtime_error("GroupNorm forward task error: " + err.message());
    }
  }

  micro_batch_inputs_[micro_batch_id] = current->clone();
  return output;
}

template <typename T>
const Tensor<T> &GroupNormLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end() || it_normalized == micro_batch_normalized_.end()) {
    throw std::runtime_error("No cached data found for micro-batch ID in GroupNormLayer: " +
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

  const Tensor<T> &input = it_input->second;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();
  const size_t spatial_size = height * width;

  Tensor<T> &grad_input = this->get_gradient_buffer(micro_batch_id, input.shape());

  auto bwd_task =
      run_backward_fused(current_gradient->data_ptr(), it_normalized->second, it_inv_std->second,
                         gamma_.data_ptr(), gamma_gradients_.data_ptr(), beta_gradients_.data_ptr(),
                         grad_input.data_ptr(), batch_size, channels, spatial_size, "default");
  if (bwd_task) {
    auto err = bwd_task->sync();
    if (err != ErrorStatus{}) {
      throw std::runtime_error("GroupNorm backward task error: " + err.message());
    }
  }

  return grad_input;
}

template <typename T>
std::unique_ptr<Task> GroupNormLayer<T>::run_forward_fused(
    const device_ptr<T[]> &input, device_ptr<T[]> &group_mean, device_ptr<T[]> &group_inv_std,
    const device_ptr<T[]> &gamma, const device_ptr<T[]> &beta, device_ptr<T[]> &output,
    device_ptr<T[]> &norm_cache, size_t batch_size, size_t channels, size_t spatial_size,
    const std::string &flow_id) {
#ifdef USE_CUDA
  if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::groupnorm::run_forward_fused<T>, input.get(),
                           group_mean.get(), group_inv_std.get(), affine_ ? gamma.get() : nullptr,
                           affine_ ? beta.get() : nullptr, output.get(), norm_cache.get(),
                           batch_size, channels, spatial_size, num_groups_, epsilon_, affine_);
  } else
#endif
  {
    return create_cpu_task("default", cpu::groupnorm::run_forward_fused<T>, input.get(),
                           group_mean.get(), group_inv_std.get(), affine_ ? gamma.get() : nullptr,
                           affine_ ? beta.get() : nullptr, output.get(), norm_cache.get(),
                           batch_size, channels, spatial_size, num_groups_, epsilon_, affine_);
  }
}

template <typename T>
std::unique_ptr<Task> GroupNormLayer<T>::run_backward_fused(
    const device_ptr<T[]> &grad_output, const device_ptr<T[]> &norm_input,
    const device_ptr<T[]> &inv_std, const device_ptr<T[]> &gamma, device_ptr<T[]> &d_gamma,
    device_ptr<T[]> &d_beta, device_ptr<T[]> &grad_input, size_t batch_size, size_t channels,
    size_t spatial_size, const std::string &flow_id) {
#ifdef USE_CUDA
  if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::groupnorm::run_backward_fused<T>, grad_output.get(),
                           norm_input.get(), inv_std.get(), gamma.get(), d_gamma.get(),
                           d_beta.get(), grad_input.get(), batch_size, channels, spatial_size,
                           num_groups_, affine_);
  } else
#endif
  {
    return create_cpu_task("default", cpu::groupnorm::run_backward_fused<T>, grad_output.get(),
                           norm_input.get(), inv_std.get(), gamma.get(), d_gamma.get(),
                           d_beta.get(), grad_input.get(), batch_size, channels, spatial_size,
                           num_groups_, affine_);
  }
}

template <typename T>
void GroupNormLayer<T>::extract_tensor_dimensions(const Tensor<T> &input, size_t &batch_size,
                                                  size_t &channels, size_t &height, size_t &width,
                                                  size_t &spatial_size) {
  batch_size = input.batch_size();
  channels = input.channels();
  height = input.height();
  width = input.width();
  spatial_size = height * width;
}

template <typename T> std::string GroupNormLayer<T>::type() const { return "groupnorm"; }

template <typename T> LayerConfig GroupNormLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["num_groups"] = num_groups_;
  config.parameters["num_channels"] = num_channels_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["affine"] = affine_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> GroupNormLayer<T>::clone() const {
  return std::make_unique<GroupNormLayer<T>>(num_groups_, num_channels_, epsilon_, affine_,
                                             this->name_);
}

template <typename T>
std::vector<size_t>
GroupNormLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T> void GroupNormLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  if (affine_) {
    params.push_back(&gamma_);
    params.push_back(&beta_);
  }
}

template <typename T> void GroupNormLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  if (affine_) {
    grads.push_back(&gamma_gradients_);
    grads.push_back(&beta_gradients_);
  }
}

template <typename T>
std::unique_ptr<Layer<T>> GroupNormLayer<T>::create_from_config(const LayerConfig &config) {
  size_t num_groups = config.get<size_t>("num_groups");
  size_t num_channels = config.get<size_t>("num_channels");
  T epsilon = config.get<T>("epsilon");
  bool affine = config.get<bool>("affine");

  return std::make_unique<GroupNormLayer<T>>(num_groups, num_channels, epsilon, affine,
                                             config.name);
}

template <typename T> void GroupNormLayer<T>::clear_gradients() {
  if (affine_) {
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));
  }
}

template <typename T>
uint64_t GroupNormLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t channels_per_group = num_channels_ / num_groups_;
  size_t spatial_size = num_elements / (batch_size * num_channels_);

  // Mean computation per group
  uint64_t mean_flops = batch_size * num_groups_ * channels_per_group * spatial_size;

  // Variance computation per group
  uint64_t var_flops = 2 * num_elements + mean_flops;

  // Normalization
  uint64_t norm_flops = 3 * num_elements;

  // Affine transformation
  uint64_t affine_flops = affine_ ? (2 * num_elements) : 0;

  return mean_flops + var_flops + norm_flops + affine_flops;
}

template <typename T>
uint64_t GroupNormLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t channels_per_group = num_channels_ / num_groups_;
  size_t spatial_size = num_elements / (batch_size * num_channels_);

  // Parameter gradients
  uint64_t param_grad_flops =
      affine_ ? (2 * batch_size * num_groups_ * channels_per_group * spatial_size) : 0;

  // Input gradients
  uint64_t input_grad_flops = 9 * num_elements;

  return param_grad_flops + input_grad_flops;
}

template <typename T>
uint64_t GroupNormLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t GroupNormLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class GroupNormLayer<float>;
template class GroupNormLayer<double>;

} // namespace tnn
