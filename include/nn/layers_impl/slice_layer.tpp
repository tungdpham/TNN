/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/task.hpp"
#include "nn/layers_impl/cpu/slice_ops.hpp"
#include "nn/layers_impl/cuda/slice_ops.hpp"
#include "nn/layers_impl/slice_layer.hpp"
#include <stdexcept>

namespace tnn {

template <typename T>
SliceLayer<T>::SliceLayer(size_t axis, size_t start, size_t length, const std::string &name)
    : StatelessLayer<T>(name), axis_(axis), start_(start), length_(length) {}

template <typename T>
void SliceLayer<T>::forward_impl(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id) {
  micro_batch_original_shapes_[micro_batch_id] = input.shape();

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

  std::vector<size_t> output_shape = compute_output_shape(current->shape());
  output.ensure(output_shape, this->device_);

  if (current->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::slice::slice_forward<T>, current->data_ptr().get(),
                    output.data_ptr().get(), current->shape(), axis_, start_, length_);
  }
#ifdef USE_CUDA
  else if (current->device_type() == DeviceType::GPU) {
    create_gpu_task("default", cuda::slice::slice_forward<T>, current->data_ptr().get(),
                    output.data_ptr().get(), current->shape(), axis_, start_, length_);
  }
#endif
  else {
    if (current->device_type() == DeviceType::GPU) {
      throw std::runtime_error("SliceLayer: GPU execution requires building with USE_CUDA");
    }
    throw std::runtime_error("SliceLayer: Unsupported device type");
  }
}

template <typename T>
void SliceLayer<T>::backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                  size_t micro_batch_id) {
  auto it = micro_batch_original_shapes_.find(micro_batch_id);
  if (it == micro_batch_original_shapes_.end()) {
    throw std::runtime_error("No cached shape found for micro-batch ID in SliceLayer");
  }
  const std::vector<size_t> &original_shape = it->second;

  const Tensor<T> *current_grad = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_grad = &device_gradient;
  }

  grad_input.ensure(original_shape, this->device_);

  if (current_grad->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::slice::slice_backward<T>, current_grad->data_ptr().get(),
                    grad_input.data_ptr().get(), original_shape, axis_, start_, length_);
  }
#ifdef USE_CUDA
  else if (current_grad->device_type() == DeviceType::GPU) {
    create_gpu_task("default", cuda::slice::slice_backward<T>, current_grad->data_ptr().get(),
                    grad_input.data_ptr().get(), original_shape, axis_, start_, length_);
  }
#endif
  else {
    if (current_grad->device_type() == DeviceType::GPU) {
      throw std::runtime_error("SliceLayer: GPU execution requires building with USE_CUDA");
    }
    throw std::runtime_error("SliceLayer: Unsupported device type");
  }
}

template <typename T>
std::vector<size_t>
SliceLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (axis_ >= input_shape.size()) {
    throw std::invalid_argument("Slice axis out of bounds");
  }
  if (start_ + length_ > input_shape[axis_]) {
    throw std::invalid_argument("Slice range out of bounds");
  }

  std::vector<size_t> output_shape = input_shape;
  output_shape[axis_] = length_;
  return output_shape;
}

template <typename T> std::string SliceLayer<T>::type() const { return "slice"; }

template <typename T> LayerConfig SliceLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["axis"] = (int)axis_;
  config.parameters["start"] = (int)start_;
  config.parameters["length"] = (int)length_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> SliceLayer<T>::clone() const {
  return std::make_unique<SliceLayer<T>>(axis_, start_, length_, this->name_);
}

template <typename T>
std::unique_ptr<Layer<T>> SliceLayer<T>::create_from_config(const LayerConfig &config) {
  size_t axis = (size_t)config.get<int>("axis", 0);
  size_t start = (size_t)config.get<int>("start", 0);
  size_t length = (size_t)config.get<int>("length", 1);
  return std::make_unique<SliceLayer<T>>(axis, start, length, config.name);
}

template <typename T>
uint64_t SliceLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template <typename T>
uint64_t SliceLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template class SliceLayer<float>;
template class SliceLayer<double>;

} // namespace tnn
