/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/avgpool2d_layer.hpp"
#include "tensor/tensor_ops.hpp"
#include <stdexcept>

#include "nn/layers_impl/cpu/avgpool_ops.hpp"
#include "nn/layers_impl/cuda/avgpool_ops.hpp"

namespace tnn {

template <typename T>
AvgPool2DLayer<T>::AvgPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
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
const Tensor<T> &AvgPool2DLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {

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

  const size_t padded_h = input_h + 2 * pad_h_;
  const size_t padded_w = input_w + 2 * pad_w_;
  const size_t output_h = (padded_h - pool_h_) / stride_h_ + 1;
  const size_t output_w = (padded_w - pool_w_) / stride_w_ + 1;

  auto it_padded = micro_batch_padded_inputs_.find(micro_batch_id);
  if (it_padded == micro_batch_padded_inputs_.end()) {
    micro_batch_padded_inputs_[micro_batch_id] =
        Tensor<T>({batch_size, channels, padded_h, padded_w}, this->device_);
    pad(*current, micro_batch_padded_inputs_[micro_batch_id], pad_h_, pad_w_);
  } else {
    it_padded->second.resize({batch_size, channels, padded_h, padded_w});
    pad(*current, it_padded->second, pad_h_, pad_w_);
  }

  Tensor<T> &output =
      this->get_output_buffer(micro_batch_id, {batch_size, channels, output_h, output_w});

  compute_avg_pool_forward(micro_batch_padded_inputs_[micro_batch_id].data_ptr(), output.data_ptr(),
                           batch_size, channels, padded_h, padded_w, output_h, output_w, "default");

  return output;
}

template <typename T>
const Tensor<T> &AvgPool2DLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input = micro_batch_padded_inputs_.find(micro_batch_id);

  if (it_input == micro_batch_padded_inputs_.end()) {
    throw std::runtime_error("No cached input found for micro-batch ID in AvgPool2DLayer: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor<T> *current_gradient = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_gradient = &device_gradient;
  }

  const Tensor<T> &cached_padded_input = it_input->second;

  const size_t batch_size = cached_padded_input.batch_size();
  const size_t channels = cached_padded_input.channels();
  const size_t padded_h = cached_padded_input.height();
  const size_t padded_w = cached_padded_input.width();
  const size_t output_h = current_gradient->height();
  const size_t output_w = current_gradient->width();

  auto it_grad_padded = micro_batch_grad_padded_inputs_.find(micro_batch_id);
  if (it_grad_padded == micro_batch_grad_padded_inputs_.end()) {
    micro_batch_grad_padded_inputs_[micro_batch_id] =
        Tensor<T>({batch_size, channels, padded_h, padded_w}, this->device_);
  } else {
    it_grad_padded->second.resize({batch_size, channels, padded_h, padded_w});
  }

  compute_avg_pool_backward(current_gradient->data_ptr(),
                            micro_batch_grad_padded_inputs_[micro_batch_id].data_ptr(), batch_size,
                            channels, padded_h, padded_w, output_h, output_w, "default");

  if (pad_h_ > 0 || pad_w_ > 0) {
    Tensor<T> &grad_input = this->get_gradient_buffer(
        micro_batch_id, {batch_size, channels, padded_h - 2 * pad_h_, padded_w - 2 * pad_w_});
    unpad(micro_batch_grad_padded_inputs_[micro_batch_id], grad_input, pad_h_, pad_w_);
    return grad_input;
  } else {
    return micro_batch_grad_padded_inputs_[micro_batch_id];
  }
}

template <typename T>
std::unique_ptr<Task> AvgPool2DLayer<T>::compute_avg_pool_forward(
    const device_ptr<T[]> &input_data, device_ptr<T[]> &output_data, size_t batch_size,
    size_t channels, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    const std::string &flow_id) const {
  if (input_data.getDeviceType() != output_data.getDeviceType()) {
    throw std::runtime_error("Input and output tensors must be on the same device");
  }

  if (input_data.getDeviceType() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::avgpool::compute_avg_pool_forward<T>, input_data.get(),
                           output_data.get(), batch_size, channels, input_h, input_w, output_h,
                           output_w, pool_h_, pool_w_, stride_h_, stride_w_);
  }
#ifdef USE_CUDA
  else if (input_data.getDeviceType() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::avgpool::compute_avg_pool_forward<T>, input_data.get(),
                           output_data.get(), batch_size, channels, input_h, input_w, output_h,
                           output_w, pool_h_, pool_w_, stride_h_, stride_w_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_avg_pool_forward");
  }
  return nullptr;
}

template <typename T>
std::unique_ptr<Task> AvgPool2DLayer<T>::compute_avg_pool_backward(
    const device_ptr<T[]> &gradient_data, device_ptr<T[]> &grad_input_data, size_t batch_size,
    size_t channels, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    const std::string &flow_id) const {
  if (gradient_data.getDeviceType() != grad_input_data.getDeviceType()) {
    throw std::runtime_error("Gradient and input gradient tensors must be on the same device");
  }

  if (gradient_data.getDeviceType() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::avgpool::compute_avg_pool_backward<T>, gradient_data.get(),
                           grad_input_data.get(), batch_size, channels, input_h, input_w, output_h,
                           output_w, pool_h_, pool_w_, stride_h_, stride_w_);
  }
#ifdef USE_CUDA
  else if (gradient_data.getDeviceType() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::avgpool::compute_avg_pool_backward<T>,
                           gradient_data.get(), grad_input_data.get(), batch_size, channels,
                           input_h, input_w, output_h, output_w, pool_h_, pool_w_, stride_h_,
                           stride_w_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_avg_pool_backward");
  }
  return nullptr;
}

template <typename T> std::string AvgPool2DLayer<T>::type() const { return "avgpool2d"; }

template <typename T> LayerConfig AvgPool2DLayer<T>::get_config() const {
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

template <typename T> std::unique_ptr<Layer<T>> AvgPool2DLayer<T>::clone() const {
  return std::make_unique<AvgPool2DLayer<T>>(pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                                             this->name_);
}

template <typename T>
std::vector<size_t>
AvgPool2DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("AvgPool2DLayer expects 4D input");
  }

  // Check for underflow in the calculation
  size_t padded_h = input_shape[2] + 2 * pad_h_;
  size_t padded_w = input_shape[3] + 2 * pad_w_;

  // Handle case where pool size is larger than input (global average pooling)
  if (padded_h < pool_h_ || padded_w < pool_w_) {
    // For global average pooling, output is 1x1
    return {input_shape[0], input_shape[1], 1, 1};
  }

  size_t output_h = (padded_h - pool_h_) / stride_h_ + 1;
  size_t output_w = (padded_w - pool_w_) / stride_w_ + 1;

  return {input_shape[0], input_shape[1], output_h, output_w};
}

template <typename T>
std::unique_ptr<Layer<T>> AvgPool2DLayer<T>::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<AvgPool2DLayer<T>>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                             config.name);
}

template <typename T>
uint64_t AvgPool2DLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  // Each output element requires pool_h * pool_w additions + 1 division
  // Approximating as 2 FLOPs per operation for addition and division
  uint64_t flops_per_output = pool_h_ * pool_w_ + 1;
  uint64_t total_outputs = batch_size * channels * output_h * output_w;

  return flops_per_output * total_outputs;
}

template <typename T>
uint64_t AvgPool2DLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];

  // Each input element receives pool_h * pool_w scaled gradient values
  // Approximating as 2 FLOPs per element (division by pool size + addition)
  uint64_t flops_per_element = 2;
  uint64_t total_inputs = batch_size * channels * input_shape[2] * input_shape[3];

  return flops_per_element * total_inputs;
}

template <typename T>
uint64_t AvgPool2DLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  // Return relative complexity for scheduling/profiling - using FLOP count as proxy
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t AvgPool2DLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  // Return relative complexity for scheduling/profiling - using FLOP count as proxy
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

// Explicit template instantiations
template class AvgPool2DLayer<float>;
template class AvgPool2DLayer<double>;

} // namespace tnn
