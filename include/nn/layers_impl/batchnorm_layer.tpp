/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/batchnorm_layer.hpp"

#include <cmath>
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
    gamma_ = Tensor<T>({num_features_, 1, 1, 1});
    beta_ = Tensor<T>({num_features_, 1, 1, 1});
    gamma_gradients_ = Tensor<T>({num_features_, 1, 1, 1});
    beta_gradients_ = Tensor<T>({num_features_, 1, 1, 1});

    gamma_.fill(T(1));
    beta_.fill(T(0));
  }

  running_mean_ = Tensor<T>({num_features_, 1, 1, 1});
  running_var_ = Tensor<T>({num_features_, 1, 1, 1});
  running_mean_gradients_ = Tensor<T>({num_features_, 1, 1, 1}); // Dummy gradients
  running_var_gradients_ = Tensor<T>({num_features_, 1, 1, 1});  // Dummy gradients

  running_mean_.fill(T(0));
  running_var_.fill(T(1));
  running_mean_gradients_.fill(T(0)); // Initialize dummy gradients to zero
  running_var_gradients_.fill(T(0));  // Initialize dummy gradients to zero

  this->initialized_ = true;
}

template <typename T>
Tensor<T> BatchNormLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != num_features_) {
    throw std::invalid_argument("Input channels must match num_features in BatchNormLayer");
  }

  micro_batch_inputs_[micro_batch_id] = input.clone();

  size_t batch_size, channels, height, width, spatial_size;
  extract_tensor_dimensions(input, batch_size, channels, height, width, spatial_size);

  Tensor<T> output(input.shape());

  if (this->is_training_) {
    Tensor<T> batch_mean({channels, 1, 1, 1});
    Tensor<T> batch_var({channels, 1, 1, 1});
    Tensor<T> batch_std({channels, 1, 1, 1});

    compute_channel_mean(input.data_ptr(), batch_mean.data_ptr(), batch_size, channels,
                         spatial_size);

    compute_channel_variance(input.data_ptr(), batch_mean.data_ptr(), batch_var.data_ptr(),
                             batch_size, channels, spatial_size);

    compute_batch_std(batch_var, batch_std, channels);

    Tensor<T> normalized(input.shape());

    if (affine_) {
      normalize_and_scale_optimized(input.data_ptr(), batch_mean.data_ptr(), batch_std.data_ptr(),
                                    gamma_.data_ptr(), beta_.data_ptr(), output.data_ptr(),
                                    normalized.data_ptr(), batch_size, channels, spatial_size,
                                    affine_);
    } else {
      device_ptr<T[]> null_ptr = nullptr;
      normalize_and_scale_optimized(input.data_ptr(), batch_mean.data_ptr(), batch_std.data_ptr(),
                                    null_ptr, null_ptr, output.data_ptr(), normalized.data_ptr(),
                                    batch_size, channels, spatial_size, affine_);
    }

    update_running_stats(batch_mean, batch_var, channels);

    micro_batch_std_[micro_batch_id] = std::move(batch_std);
    micro_batch_normalized_[micro_batch_id] = std::move(normalized);

  } else {
    compute_inference_output(input, output, batch_size, channels, spatial_size);
  }

  return output;
}

template <typename T>
void BatchNormLayer<T>::forward_inplace(Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != num_features_) {
    throw std::invalid_argument("Input channels must match num_features in BatchNormLayer");
  }

  size_t batch_size, channels, height, width, spatial_size;
  extract_tensor_dimensions(input, batch_size, channels, height, width, spatial_size);

  Tensor<T> output(input.shape());

  if (this->is_training_) {
    Tensor<T> batch_mean({channels, 1, 1, 1});
    Tensor<T> batch_var({channels, 1, 1, 1});
    Tensor<T> batch_std({channels, 1, 1, 1});

    compute_channel_mean(input.data_ptr(), batch_mean.data_ptr(), batch_size, channels,
                         spatial_size);

    compute_channel_variance(input.data_ptr(), batch_mean.data_ptr(), batch_var.data_ptr(),
                             batch_size, channels, spatial_size);

    compute_batch_std(batch_var, batch_std, channels);

    Tensor<T> normalized(input.shape());

    if (affine_) {
      normalize_and_scale_optimized(input.data_ptr(), batch_mean.data_ptr(), batch_std.data_ptr(),
                                    gamma_.data_ptr(), beta_.data_ptr(), output.data_ptr(),
                                    normalized.data_ptr(), batch_size, channels, spatial_size,
                                    affine_);
    } else {
      device_ptr<T[]> null_ptr = nullptr;
      normalize_and_scale_optimized(input.data_ptr(), batch_mean.data_ptr(), batch_std.data_ptr(),
                                    null_ptr, null_ptr, output.data_ptr(), normalized.data_ptr(),
                                    batch_size, channels, spatial_size, affine_);
    }

    update_running_stats(batch_mean, batch_var, channels);

    micro_batch_std_[micro_batch_id] = std::move(batch_std);
    micro_batch_normalized_[micro_batch_id] = std::move(normalized);

  } else {
    compute_inference_output(input, output, batch_size, channels, spatial_size);
  }

  micro_batch_inputs_[micro_batch_id] = std::move(input);
  input = std::move(output);
}

template <typename T>
Tensor<T> BatchNormLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);
  auto it_std = micro_batch_std_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end() || it_normalized == micro_batch_normalized_.end() ||
      it_std == micro_batch_std_.end()) {
    throw std::runtime_error("No cached data found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor<T> &input = it_input->second;
  const Tensor<T> &normalized = it_normalized->second;
  const Tensor<T> &std_val = it_std->second;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();
  const size_t spatial_size = height * width;
  const size_t total_elements = batch_size * spatial_size;

  Tensor<T> grad_input(input.shape());

  if (affine_) {
    compute_affine_gradients_optimized(gradient.data_ptr(), normalized.data_ptr(),
                                       gamma_gradients_.data_ptr(), beta_gradients_.data_ptr(),
                                       batch_size, channels, spatial_size);
  }

  Tensor<T> grad_normalized(input.shape());

  // Compute grad_normalized using wrapper function
  Tensor<T> dummy_gamma; // Empty tensor for non-affine case
  compute_grad_normalized_wrapper(
      gradient.data_ptr(), affine_ ? gamma_.data_ptr() : dummy_gamma.data_ptr(),
      grad_normalized.data_ptr(), batch_size, channels, spatial_size, affine_);

  Tensor<T> sum_grad_normalized({channels, 1, 1, 1});
  Tensor<T> sum_grad_normalized_times_normalized({channels, 1, 1, 1});

  // Compute backward sums using wrapper function
  compute_backward_sums_wrapper(
      grad_normalized.data_ptr(), normalized.data_ptr(), sum_grad_normalized.data_ptr(),
      sum_grad_normalized_times_normalized.data_ptr(), batch_size, channels, spatial_size);

  // Compute input gradients using wrapper function
  compute_input_gradients_batchnorm_wrapper(
      grad_normalized.data_ptr(), normalized.data_ptr(), std_val.data_ptr(),
      sum_grad_normalized.data_ptr(), sum_grad_normalized_times_normalized.data_ptr(),
      grad_input.data_ptr(), batch_size, channels, spatial_size, total_elements);

  return grad_input;
}

template <typename T>
void BatchNormLayer<T>::compute_batch_std(const Tensor<T> &batch_var, Tensor<T> &batch_std,
                                          size_t channels) {
  if (batch_var.data_ptr().getDeviceType() != batch_std.data_ptr().getDeviceType()) {
    throw std::runtime_error("Batch variance and std tensors must be on the same device");
  }

  if (batch_var.data_ptr().getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_batch_std(batch_var.data_ptr().get(), batch_std.data_ptr().get(),
                                      channels, epsilon_);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_batch_std(batch_var.data_ptr().get(), batch_std.data_ptr().get(),
                                       channels, epsilon_);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::update_running_stats(const Tensor<T> &batch_mean,
                                             const Tensor<T> &batch_var, size_t channels) {
  if (running_mean_.data_ptr().getDeviceType() != running_var_.data_ptr().getDeviceType() ||
      running_mean_.data_ptr().getDeviceType() != batch_mean.data_ptr().getDeviceType() ||
      batch_mean.data_ptr().getDeviceType() != batch_var.data_ptr().getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for update_running_stats");
  }

  if (running_mean_.data_ptr().getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::update_running_stats(running_mean_.data_ptr().get(),
                                         running_var_.data_ptr().get(), batch_mean.data_ptr().get(),
                                         batch_var.data_ptr().get(), channels, momentum_);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::update_running_stats(
        running_mean_.data_ptr().get(), running_var_.data_ptr().get(), batch_mean.data_ptr().get(),
        batch_var.data_ptr().get(), channels, momentum_);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_inference_output(const Tensor<T> &input, Tensor<T> &output,
                                                 size_t batch_size, size_t channels,
                                                 size_t spatial_size) {
  if (input.data_ptr().getDeviceType() != output.data_ptr().getDeviceType() ||
      input.data_ptr().getDeviceType() != running_mean_.data_ptr().getDeviceType() ||
      running_mean_.data_ptr().getDeviceType() != running_var_.data_ptr().getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for inference output");
  }

  if (affine_ && (input.data_ptr().getDeviceType() != gamma_.data_ptr().getDeviceType() ||
                  gamma_.data_ptr().getDeviceType() != beta_.data_ptr().getDeviceType())) {
    throw std::runtime_error("Gamma and beta must be on the same device as input");
  }

  if (input.data_ptr().getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_inference_output(
        input.data_ptr().get(), running_mean_.data_ptr().get(), running_var_.data_ptr().get(),
        affine_ ? gamma_.data_ptr().get() : nullptr, affine_ ? beta_.data_ptr().get() : nullptr,
        output.data_ptr().get(), batch_size, channels, spatial_size, epsilon_, affine_);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_inference_output(
        input.data_ptr().get(), running_mean_.data_ptr().get(), running_var_.data_ptr().get(),
        affine_ ? gamma_.data_ptr().get() : nullptr, affine_ ? beta_.data_ptr().get() : nullptr,
        output.data_ptr().get(), batch_size, channels, spatial_size, epsilon_, affine_);
  }
#endif
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
template <typename T>
void BatchNormLayer<T>::compute_channel_mean(const device_ptr<T[]> &input_data,
                                             device_ptr<T[]> &mean_data, size_t batch_size,
                                             size_t channels, size_t spatial_size) {
  if (input_data.getDeviceType() != mean_data.getDeviceType()) {
    throw std::runtime_error("Input and mean tensors must be on the same device");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_channel_mean(input_data.get(), mean_data.get(), batch_size, channels,
                                         spatial_size);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_channel_mean(input_data.get(), mean_data.get(), batch_size, channels,
                                          spatial_size);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_channel_variance(const device_ptr<T[]> &input_data,
                                                 const device_ptr<T[]> &mean_data,
                                                 device_ptr<T[]> &var_data, size_t batch_size,
                                                 size_t channels, size_t spatial_size) {
  if (input_data.getDeviceType() != mean_data.getDeviceType() ||
      mean_data.getDeviceType() != var_data.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for channel variance");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_channel_variance(input_data.get(), mean_data.get(), var_data.get(),
                                             batch_size, channels, spatial_size);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_channel_variance(input_data.get(), mean_data.get(), var_data.get(),
                                              batch_size, channels, spatial_size);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::normalize_and_scale_optimized(
    const device_ptr<T[]> &input_data, const device_ptr<T[]> &mean_data,
    const device_ptr<T[]> &std_data, const device_ptr<T[]> &gamma_data,
    const device_ptr<T[]> &beta_data, device_ptr<T[]> &output_data,
    device_ptr<T[]> &normalized_data, size_t batch_size, size_t channels, size_t spatial_size,
    bool affine) {
  if (input_data.getDeviceType() != mean_data.getDeviceType() ||
      mean_data.getDeviceType() != std_data.getDeviceType() ||
      std_data.getDeviceType() != output_data.getDeviceType() ||
      output_data.getDeviceType() != normalized_data.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for normalize and scale");
  }

  if (affine && (gamma_data.getDeviceType() != input_data.getDeviceType() ||
                 beta_data.getDeviceType() != input_data.getDeviceType())) {
    throw std::runtime_error("Gamma and beta tensors must be on the same device as input");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::normalize_and_scale_optimized(
        input_data.get(), mean_data.get(), std_data.get(), affine ? gamma_data.get() : nullptr,
        affine ? beta_data.get() : nullptr, output_data.get(), normalized_data.get(), batch_size,
        channels, spatial_size, affine);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::normalize_and_scale_optimized(
        input_data.get(), mean_data.get(), std_data.get(), affine ? gamma_data.get() : nullptr,
        affine ? beta_data.get() : nullptr, output_data.get(), normalized_data.get(), batch_size,
        channels, spatial_size, affine);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_affine_gradients_optimized(const device_ptr<T[]> &gradient_data,
                                                           const device_ptr<T[]> &normalized_data,
                                                           device_ptr<T[]> &gamma_grad,
                                                           device_ptr<T[]> &beta_grad,
                                                           size_t batch_size, size_t channels,
                                                           size_t spatial_size) {
  if (gradient_data.getDeviceType() != normalized_data.getDeviceType() ||
      normalized_data.getDeviceType() != gamma_grad.getDeviceType() ||
      gamma_grad.getDeviceType() != beta_grad.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for affine gradients");
  }

  if (gradient_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_affine_gradients_optimized(gradient_data.get(), normalized_data.get(),
                                                       gamma_grad.get(), beta_grad.get(),
                                                       batch_size, channels, spatial_size);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_affine_gradients_optimized(gradient_data.get(), normalized_data.get(),
                                                        gamma_grad.get(), beta_grad.get(),
                                                        batch_size, channels, spatial_size);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_batch_std_wrapper(const device_ptr<T[]> &batch_var_data,
                                                  device_ptr<T[]> &batch_std_data, size_t channels,
                                                  T epsilon) {
  if (batch_var_data.getDeviceType() != batch_std_data.getDeviceType()) {
    throw std::runtime_error("Batch variance and std tensors must be on the same device");
  }

  if (batch_var_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_batch_std(batch_var_data.get(), batch_std_data.get(), channels,
                                      epsilon);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_batch_std(batch_var_data.get(), batch_std_data.get(), channels,
                                       epsilon);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::update_running_stats_wrapper(device_ptr<T[]> &running_mean_data,
                                                     device_ptr<T[]> &running_var_data,
                                                     const device_ptr<T[]> &batch_mean_data,
                                                     const device_ptr<T[]> &batch_var_data,
                                                     size_t channels, T momentum) {
  if (running_mean_data.getDeviceType() != running_var_data.getDeviceType() ||
      running_mean_data.getDeviceType() != batch_mean_data.getDeviceType() ||
      batch_mean_data.getDeviceType() != batch_var_data.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for update_running_stats");
  }

  if (running_mean_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::update_running_stats(running_mean_data.get(), running_var_data.get(),
                                         batch_mean_data.get(), batch_var_data.get(), channels,
                                         momentum);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::update_running_stats(running_mean_data.get(), running_var_data.get(),
                                          batch_mean_data.get(), batch_var_data.get(), channels,
                                          momentum);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_inference_output_wrapper(
    const device_ptr<T[]> &input_data, const device_ptr<T[]> &running_mean_data,
    const device_ptr<T[]> &running_var_data, const device_ptr<T[]> &gamma_data,
    const device_ptr<T[]> &beta_data, device_ptr<T[]> &output_data, size_t batch_size,
    size_t channels, size_t spatial_size, T epsilon, bool affine) {
  if (input_data.getDeviceType() != output_data.getDeviceType() ||
      input_data.getDeviceType() != running_mean_data.getDeviceType() ||
      running_mean_data.getDeviceType() != running_var_data.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for inference output");
  }

  if (affine && (input_data.getDeviceType() != gamma_data.getDeviceType() ||
                 gamma_data.getDeviceType() != beta_data.getDeviceType())) {
    throw std::runtime_error("Gamma and beta must be on the same device as input");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_inference_output(
        input_data.get(), running_mean_data.get(), running_var_data.get(),
        affine ? gamma_data.get() : nullptr, affine ? beta_data.get() : nullptr, output_data.get(),
        batch_size, channels, spatial_size, epsilon, affine);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_inference_output(
        input_data.get(), running_mean_data.get(), running_var_data.get(),
        affine ? gamma_data.get() : nullptr, affine ? beta_data.get() : nullptr, output_data.get(),
        batch_size, channels, spatial_size, epsilon, affine);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_grad_normalized_wrapper(const device_ptr<T[]> &gradient_data,
                                                        const device_ptr<T[]> &gamma_data,
                                                        device_ptr<T[]> &grad_normalized_data,
                                                        size_t batch_size, size_t channels,
                                                        size_t spatial_size, bool affine) {
  if (gradient_data.getDeviceType() != grad_normalized_data.getDeviceType()) {
    throw std::runtime_error("Gradient tensors must be on the same device");
  }

  if (affine && gamma_data.getDeviceType() != gradient_data.getDeviceType()) {
    throw std::runtime_error("Gamma must be on the same device as gradient");
  }

  if (gradient_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_grad_normalized(
        gradient_data.get(), affine ? gamma_data.get() : nullptr, grad_normalized_data.get(),
        batch_size, channels, spatial_size, affine);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_grad_normalized(
        gradient_data.get(), affine ? gamma_data.get() : nullptr, grad_normalized_data.get(),
        batch_size, channels, spatial_size, affine);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_backward_sums_wrapper(
    const device_ptr<T[]> &grad_normalized_data, const device_ptr<T[]> &normalized_data,
    device_ptr<T[]> &sum_grad_normalized_data, device_ptr<T[]> &sum_grad_norm_times_norm_data,
    size_t batch_size, size_t channels, size_t spatial_size) {
  if (grad_normalized_data.getDeviceType() != normalized_data.getDeviceType() ||
      normalized_data.getDeviceType() != sum_grad_normalized_data.getDeviceType() ||
      sum_grad_normalized_data.getDeviceType() != sum_grad_norm_times_norm_data.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for backward sums");
  }

  if (grad_normalized_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_backward_sums(
        grad_normalized_data.get(), normalized_data.get(), sum_grad_normalized_data.get(),
        sum_grad_norm_times_norm_data.get(), batch_size, channels, spatial_size);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_backward_sums(
        grad_normalized_data.get(), normalized_data.get(), sum_grad_normalized_data.get(),
        sum_grad_norm_times_norm_data.get(), batch_size, channels, spatial_size);
  }
#endif
}

template <typename T>
void BatchNormLayer<T>::compute_input_gradients_batchnorm_wrapper(
    const device_ptr<T[]> &grad_normalized_data, const device_ptr<T[]> &normalized_data,
    const device_ptr<T[]> &std_data, const device_ptr<T[]> &sum_grad_normalized_data,
    const device_ptr<T[]> &sum_grad_norm_times_norm_data, device_ptr<T[]> &grad_input_data,
    size_t batch_size, size_t channels, size_t spatial_size, size_t total_elements) {
  if (grad_normalized_data.getDeviceType() != normalized_data.getDeviceType() ||
      normalized_data.getDeviceType() != std_data.getDeviceType() ||
      std_data.getDeviceType() != sum_grad_normalized_data.getDeviceType() ||
      sum_grad_normalized_data.getDeviceType() != sum_grad_norm_times_norm_data.getDeviceType() ||
      sum_grad_norm_times_norm_data.getDeviceType() != grad_input_data.getDeviceType()) {
    throw std::runtime_error("All tensors must be on the same device for input gradients");
  }

  if (grad_normalized_data.getDeviceType() == DeviceType::CPU) {
    cpu::batchnorm::compute_input_gradients_batchnorm(
        grad_normalized_data.get(), normalized_data.get(), std_data.get(),
        sum_grad_normalized_data.get(), sum_grad_norm_times_norm_data.get(), grad_input_data.get(),
        batch_size, channels, spatial_size, total_elements);
  }
#ifdef USE_CUDA
  else {
    cuda::batchnorm::compute_input_gradients_batchnorm(
        grad_normalized_data.get(), normalized_data.get(), std_data.get(),
        sum_grad_normalized_data.get(), sum_grad_norm_times_norm_data.get(), grad_input_data.get(),
        batch_size, channels, spatial_size, total_elements);
  }
#endif
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

  // mean computation: sum reduction across batch and spatial dimensions
  uint64_t mean_flops = batch_size * spatial_size * num_features_;

  // variance computation: (x - mean)^2 for each element + sum reduction
  uint64_t var_flops = 2 * num_elements + mean_flops;

  // normalization: (x - mean) / sqrt(var + epsilon) for each element
  uint64_t norm_flops = 3 * num_elements; // subtract, sqrt+add, divide

  // scale and shift (if affine): gamma * normalized + beta
  uint64_t affine_flops = affine_ ? (2 * num_elements) : 0;

  return mean_flops + var_flops + norm_flops + affine_flops;
}

template <typename T>
uint64_t BatchNormLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  // Gradient w.r.t. gamma and beta (if affine)
  uint64_t param_grad_flops = affine_ ? (2 * batch_size * spatial_size * num_features_) : 0;

  // Gradient w.r.t. input involves multiple terms:
  // 1. Direct gradient scaling: 2 * num_elements
  // 2. Mean gradient compensation: 3 * num_elements
  // 3. Variance gradient compensation: 4 * num_elements
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

// Explicit template instantiations
template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

} // namespace tnn
