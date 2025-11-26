/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/task.hpp"
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include <stdexcept>

#include "nn/layers_impl/cpu/maxpool_ops.hpp"
#include "nn/layers_impl/cuda/maxpool_ops.hpp"

namespace tnn {

template <typename T>
MaxPool2DLayer<T>::MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                  size_t pad_h, size_t pad_w, const std::string &name)
    : StatelessLayer<T>(name), pool_h_(pool_h), pool_w_(pool_w),
      stride_h_(stride_h == 0 ? pool_h : stride_h), stride_w_(stride_w == 0 ? pool_w : stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {

  if (pool_h_ == 0 || pool_w_ == 0) {
    throw std::invalid_argument("Pool dimensions must be positive");
  }
  if (stride_h_ == 0 || stride_w_ == 0) {
    throw std::invalid_argument("Stride dimensions must be positive");
  }
}

template <typename T>
const Tensor<T> &MaxPool2DLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  const size_t batch_size = current->batch_size();
  const size_t channels = current->channels();
  const size_t input_h = current->height();
  const size_t input_w = current->width();

  // Cache input shape for backward pass
  micro_batch_input_shapes_[micro_batch_id] = {batch_size, channels, input_h, input_w};

  // Calculate output shape directly from input (implicit padding)
  const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  Tensor<T> &output = this->get_buffer({batch_size, channels, output_h, output_w});

  const size_t total_outputs = batch_size * channels * output_h * output_w;

  auto it_mask = micro_batch_mask_indices_.find(micro_batch_id);
  if (it_mask == micro_batch_mask_indices_.end()) {
    micro_batch_mask_indices_[micro_batch_id] =
        make_array_ptr<size_t[]>(this->device_, total_outputs);
  }

  // Pass raw input pointer + padding params (implicit padding)
  forward_task_ = compute_max_pool_forward(current->data_ptr(), output.data_ptr(), batch_size,
                                           channels, input_h, input_w, output_h, output_w,
                                           micro_batch_mask_indices_[micro_batch_id], "default");

  return output;
}

template <typename T>
const Tensor<T> &MaxPool2DLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_mask = micro_batch_mask_indices_.find(micro_batch_id);
  auto it_shape = micro_batch_input_shapes_.find(micro_batch_id);

  if (it_mask == micro_batch_mask_indices_.end()) {
    throw std::runtime_error("No cached mask found for micro-batch ID in MaxPool2DLayer: " +
                             std::to_string(micro_batch_id));
  }
  if (it_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID in MaxPool2DLayer: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor<T> *current_gradient = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_gradient = &device_gradient;
  }

  const device_ptr<size_t[]> &mask_indices = it_mask->second;
  const std::vector<size_t> &input_shape = it_shape->second;

  const size_t batch_size = input_shape[0];
  const size_t channels = input_shape[1];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = current_gradient->height();
  const size_t output_w = current_gradient->width();

  Tensor<T> &grad_input = this->get_buffer({batch_size, channels, input_h, input_w});

  grad_input.fill(T(0));

  backward_task_ =
      compute_max_pool_backward(current_gradient->data_ptr(), grad_input.data_ptr(), batch_size,
                                channels, output_h, output_w, mask_indices, "default");

  return grad_input;
}

template <typename T>
std::unique_ptr<Task> MaxPool2DLayer<T>::compute_max_pool_forward(
    const device_ptr<T[]> &input_data, device_ptr<T[]> &output_data, size_t batch_size,
    size_t channels, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    device_ptr<size_t[]> &mask_indices, const std::string &flow_id) const {
  if (input_data.device_type() != output_data.device_type()) {
    throw std::runtime_error("Input and output tensors must be on the same device");
  }

  if (input_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::maxpool::compute_max_pool_forward<T>, input_data.get(),
                           output_data.get(), batch_size, channels, input_h, input_w, output_h,
                           output_w, pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                           mask_indices);
  }
#ifdef USE_CUDA
  else if (input_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::maxpool::compute_max_pool_forward<T>, input_data.get(),
                           output_data.get(), batch_size, channels, input_h, input_w, output_h,
                           output_w, pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                           mask_indices);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_max_pool_forward");
  }
  return nullptr;
}

template <typename T>
std::unique_ptr<Task> MaxPool2DLayer<T>::compute_max_pool_backward(
    const device_ptr<T[]> &gradient_data, device_ptr<T[]> &grad_input_data, size_t batch_size,
    size_t channels, size_t output_h, size_t output_w, const device_ptr<size_t[]> &mask_indices,
    const std::string &flow_id) const {
  if (gradient_data.device_type() != grad_input_data.device_type()) {
    throw std::runtime_error("Gradient and input gradient tensors must be on the same device");
  }

  if (gradient_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::maxpool::compute_max_pool_backward<T>, gradient_data.get(),
                           grad_input_data.get(), batch_size, channels, output_h, output_w,
                           mask_indices);
  }
#ifdef USE_CUDA
  else if (gradient_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::maxpool::compute_max_pool_backward<T>,
                           gradient_data.get(), grad_input_data.get(), batch_size, channels,
                           output_h, output_w, mask_indices);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_max_pool_backward");
  }
  return nullptr;
}

template <typename T> std::string MaxPool2DLayer<T>::type() const { return "maxpool2d"; }

template <typename T> LayerConfig MaxPool2DLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["pool_h"] = pool_h_;
  config.parameters["pool_w"] = pool_w_;
  config.parameters["stride_h"] = stride_h_;
  config.parameters["stride_w"] = stride_w_;
  config.parameters["pad_h"] = pad_h_;
  config.parameters["pad_w"] = pad_w_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> MaxPool2DLayer<T>::clone() const {
  return std::make_unique<MaxPool2DLayer<T>>(pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                                             this->name_);
}

template <typename T>
std::vector<size_t>
MaxPool2DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2DLayer expects 4D input");
  }

  size_t output_h = (input_shape[2] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  return {input_shape[0], input_shape[1], output_h, output_w};
}

template <typename T>
std::unique_ptr<Layer<T>> MaxPool2DLayer<T>::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<MaxPool2DLayer<T>>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                             config.name);
}

template <typename T>
uint64_t MaxPool2DLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  // Each output element requires pool_h * pool_w comparisons to find max
  // Approximating comparisons as 1 FLOP each
  uint64_t comparisons_per_output = pool_h_ * pool_w_;
  uint64_t total_outputs = batch_size * channels * output_h * output_w;

  return comparisons_per_output * total_outputs;
}

template <typename T>
uint64_t MaxPool2DLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  // Each output gradient element gets routed to exactly one input position
  // This is essentially a scatter operation with minimal computation
  return batch_size * channels * output_h * output_w;
}

template <typename T>
uint64_t MaxPool2DLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  // Return relative complexity for scheduling/profiling - using FLOP count as proxy
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t MaxPool2DLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  // Return relative complexity for scheduling/profiling - using FLOP count as proxy
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

// Explicit template instantiations
template class MaxPool2DLayer<float>;
template class MaxPool2DLayer<double>;

} // namespace tnn
