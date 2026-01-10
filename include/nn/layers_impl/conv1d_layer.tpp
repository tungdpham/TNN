/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/layers_impl/conv1d_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/conv2d_ops.hpp"
#include "nn/layers_impl/cuda/conv2d_ops.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor_ops.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tnn {

template <typename T>
Conv1DLayer<T>::Conv1DLayer(size_t in_channels, size_t out_channels, size_t kernel_size,
                            size_t stride, size_t padding, bool use_bias, const std::string &name)
    : ParameterizedLayer<T>(name), in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias) {}

template <typename T> Conv1DLayer<T>::~Conv1DLayer() {}

template <typename T> void Conv1DLayer<T>::initialize_params() {
  weights_ = Tensor<T>({out_channels_, in_channels_, 1, kernel_size_}, this->device_);
  weight_gradients_ = Tensor<T>({out_channels_, in_channels_, 1, kernel_size_}, this->device_);
  weight_gradients_.fill(T(0));

  if (use_bias_) {
    bias_ = Tensor<T>({out_channels_, 1, 1, 1}, this->device_);
    bias_gradients_ = Tensor<T>({out_channels_, 1, 1, 1}, this->device_);
    bias_.fill(T(0));
    bias_gradients_.fill(T(0));
  }

  temp_output_buffer_ = make_array_ptr<T[]>(this->device_, 0);
  temp_gradient_buffer_ = make_array_ptr<T[]>(this->device_, 0);
  temp_col_grad_matrix_buffer_ = make_array_ptr<T[]>(this->device_, 0);

  T fan_in = static_cast<T>(in_channels_ * kernel_size_);
  T bound = static_cast<T>(1.0) / std::sqrt(fan_in);

  if (this->use_seed_) {
    weights_.fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    weights_.fill_random_uniform(-bound, bound);
  }

  if (use_bias_) {
    if (this->use_seed_) {
      bias_.fill_random_uniform(-bound, bound, this->srand_seed_);
    } else {
      bias_.fill_random_uniform(-bound, bound);
    }
  }
}

template <typename T>
void Conv1DLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {
  if (!this->initialized_) {
    throw std::runtime_error("Conv1DLayer must be initialized before forward pass.");
  }

  const auto &shape = input.shape();
  if (shape.size() != 3) {
    throw std::invalid_argument("Conv1D: Input tensor must be 3-dimensional (NCL)");
  }

  size_t batch_size = shape[0];
  size_t in_channels = shape[1];
  size_t input_len = shape[2];

  if (in_channels != in_channels_) {
    throw std::invalid_argument("Conv1D: Input channels mismatch");
  }

  size_t output_len = (input_len + 2 * padding_ - kernel_size_) / stride_ + 1;
  output.ensure({batch_size, out_channels_, output_len}, this->device_);

  micro_batch_input_shapes_[micro_batch_id] = shape;

  size_t kernel_elements = in_channels_ * kernel_size_;
  size_t output_size = batch_size * output_len;
  size_t col_matrix_size = kernel_elements * output_size;

  auto &col_buffer = micro_batch_col_buffers_[micro_batch_id];
  if (!col_buffer || col_buffer.size() < col_matrix_size) {
    col_buffer = make_array_ptr<T[]>(this->device_, col_matrix_size);
  }

  temp_output_buffer_.ensure(out_channels_ * output_size);

  // Use im2col with H=1
  im2col_task_ = im2col(input, col_buffer, 1, kernel_size_, 1, stride_, 0, padding_);

  forward_task_ = compute_conv_forward(col_buffer, weights_.data_ptr(), temp_output_buffer_,
                                       output_size, kernel_elements, out_channels_, "default");

  // Output from forward is CNHW-like, where H=1. We need [N, C, L].
  // But wait, compute_conv_forward returns [C, N*L].
  // nchw_to_cnhw and back should handle it.
  // Actually, ops::cnhw_to_nchw works for [C, N, 1, W] -> [N, C, 1, W] which is [N, C, W].
  ops::cnhw_to_nchw(temp_output_buffer_, output.data_ptr(), batch_size, out_channels_, 1,
                    output_len);

  if (use_bias_) {
    add_bias_task_ = add_bias_vector(output.data_ptr(), bias_.data_ptr(), batch_size, output_len,
                                     out_channels_, "default");
  }
}

template <typename T>
void Conv1DLayer<T>::backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                              size_t micro_batch_id) {
  const auto &input_shape = micro_batch_input_shapes_[micro_batch_id];
  size_t batch_size = input_shape[0];
  size_t input_len = input_shape[2];
  size_t output_len = gradient.shape()[2];

  grad_input.ensure(input_shape, this->device_);
  grad_input.fill(T(0));

  auto &col_buffer = micro_batch_col_buffers_[micro_batch_id];
  size_t kernel_elements = in_channels_ * kernel_size_;
  size_t output_size = batch_size * output_len;
  size_t col_grad_matrix_size = kernel_elements * output_size;

  temp_gradient_buffer_.ensure(out_channels_ * output_size);
  temp_col_grad_matrix_buffer_.ensure(col_grad_matrix_size);

  ops::nchw_to_cnhw(gradient.data_ptr(), temp_gradient_buffer_, batch_size, out_channels_, 1,
                    output_len, "default");

  weight_grad_task_ =
      compute_weight_gradients(col_buffer, temp_gradient_buffer_, weight_gradients_.data_ptr(),
                               output_size, kernel_elements, out_channels_, "default");

  input_grad_task_ = compute_input_gradients(temp_gradient_buffer_, weights_.data_ptr(),
                                             temp_col_grad_matrix_buffer_, output_size,
                                             kernel_elements, out_channels_, "default");

  col2im_task_ = col2im(temp_col_grad_matrix_buffer_, grad_input.data_ptr(), batch_size,
                        in_channels_, 1, input_len, 1, kernel_size_, 1, stride_, 0, padding_);

  if (use_bias_) {
    bias_grad_task_ = compute_bias_gradients(gradient.data_ptr(), bias_gradients_.data_ptr(),
                                             batch_size, output_len, out_channels_, "default");
  }
}

template <typename T>
std::unique_ptr<Task> Conv1DLayer<T>::compute_conv_forward(
    const device_ptr<T[]> &col_data, const device_ptr<T[]> &weight_data,
    device_ptr<T[]> &output_data, const size_t output_size, const size_t kernel_size,
    const size_t out_channels, const std::string &flow_id) {
  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_conv_forward<T>, col_data.get(),
                           weight_data.get(), output_data.get(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_conv_forward<T>, col_data.get(),
                           weight_data.get(), output_data.get(), output_size, kernel_size,
                           out_channels);
  }
#endif
  return nullptr;
}

template <typename T>
std::unique_ptr<Task> Conv1DLayer<T>::compute_weight_gradients(
    const device_ptr<T[]> &col_data, const device_ptr<T[]> &gradient_data,
    device_ptr<T[]> &weight_grad_data, const size_t output_size, const size_t kernel_size,
    const size_t out_channels, const std::string &flow_id) {
  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_weight_gradients<T>, col_data.get(),
                           gradient_data.get(), weight_grad_data.get(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_weight_gradients<T>, col_data.get(),
                           gradient_data.get(), weight_grad_data.get(), output_size, kernel_size,
                           out_channels);
  }
#endif
  return nullptr;
}

template <typename T>
std::unique_ptr<Task> Conv1DLayer<T>::compute_input_gradients(
    const device_ptr<T[]> &gradient_data, const device_ptr<T[]> &weight_data,
    device_ptr<T[]> &grad_input_col_data, const size_t output_size, const size_t kernel_size,
    const size_t out_channels, const std::string &flow_id) const {
  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_input_gradients<T>, gradient_data.get(),
                           weight_data.get(), grad_input_col_data.get(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_input_gradients<T>, gradient_data.get(),
                           weight_data.get(), grad_input_col_data.get(), output_size, kernel_size,
                           out_channels);
  }
#endif
  return nullptr;
}

template <typename T>
std::unique_ptr<Task> Conv1DLayer<T>::compute_bias_gradients(
    const device_ptr<T[]> &gradient_data, device_ptr<T[]> &bias_grad_data, const size_t batch_size,
    const size_t output_len, const size_t out_channels, const std::string &flow_id) const {
  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_bias_gradients<T>, gradient_data.get(),
                           bias_grad_data.get(), batch_size, 1, output_len, out_channels);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_bias_gradients<T>, gradient_data.get(),
                           bias_grad_data.get(), batch_size, 1, output_len, out_channels);
  }
#endif
  return nullptr;
}

template <typename T>
std::unique_ptr<Task>
Conv1DLayer<T>::add_bias_vector(device_ptr<T[]> &output_data, const device_ptr<T[]> &bias_data,
                                const size_t batch_size, const size_t output_len,
                                const size_t out_channels, const std::string &flow_id) const {
  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::add_bias_to_output<T>, output_data.get(),
                           bias_data.get(), batch_size, 1, output_len, out_channels);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::add_bias_to_output<T>, output_data.get(),
                           bias_data.get(), batch_size, 1, output_len, out_channels);
  }
#endif
  return nullptr;
}

template <typename T>
std::vector<size_t>
Conv1DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 3) {
    throw std::invalid_argument("Conv1D compute_output_shape: input must be [N, C, L]");
  }
  size_t output_len = (input_shape[2] + 2 * padding_ - kernel_size_) / stride_ + 1;
  return {input_shape[0], out_channels_, output_len};
}

template <typename T>
uint64_t Conv1DLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  size_t output_len = (input_shape[2] + 2 * padding_ - kernel_size_) / stride_ + 1;
  return 2 * (uint64_t)batch_size * out_channels_ * in_channels_ * kernel_size_ * output_len;
}

template <typename T>
uint64_t Conv1DLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  return 3 * forward_complexity(input_shape);
}

template <typename T> LayerConfig Conv1DLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["in_channels"] = in_channels_;
  config.parameters["out_channels"] = out_channels_;
  config.parameters["kernel_size"] = kernel_size_;
  config.parameters["stride"] = stride_;
  config.parameters["padding"] = padding_;
  config.parameters["use_bias"] = use_bias_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> Conv1DLayer<T>::clone() const {
  return std::make_unique<Conv1DLayer<T>>(in_channels_, out_channels_, kernel_size_, stride_,
                                          padding_, use_bias_, this->name_);
}

template <typename T> void Conv1DLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_)
    params.push_back(&bias_);
}

template <typename T> void Conv1DLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_)
    grads.push_back(&bias_gradients_);
}

template <typename T> void Conv1DLayer<T>::clear_gradients() {
  if (this->initialized_) {
    weight_gradients_.fill(T(0));
    if (use_bias_)
      bias_gradients_.fill(T(0));
  }
}

template <typename T> size_t Conv1DLayer<T>::cached_memory_bytes() const {
  size_t total = 0;
  for (const auto &kv : micro_batch_col_buffers_) {
    total += kv.second.capacity() * sizeof(T);
  }
  return total;
}

template <typename T>
std::unique_ptr<Layer<T>> Conv1DLayer<T>::create_from_config(const LayerConfig &config) {
  size_t in_channels = config.get<size_t>("in_channels");
  size_t out_channels = config.get<size_t>("out_channels");
  size_t kernel_size = config.get<size_t>("kernel_size");
  size_t stride = config.get<size_t>("stride", 1);
  size_t padding = config.get<size_t>("padding", 0);
  bool use_bias = config.get<bool>("use_bias", true);

  return std::make_unique<Conv1DLayer<T>>(in_channels, out_channels, kernel_size, stride, padding,
                                          use_bias, config.name);
}

} // namespace tnn
