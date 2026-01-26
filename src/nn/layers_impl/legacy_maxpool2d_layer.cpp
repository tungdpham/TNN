/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_maxpool2d_layer.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/maxpool_nchw_ops.hpp"
#include "nn/layers_impl/cuda/maxpool_nchw_ops.hpp"
#include "tensor/tensor.hpp"

#include <cstddef>
#include <stdexcept>

namespace tnn {

LegacyMaxPool2DLayer::LegacyMaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h,
                                           size_t stride_w, size_t pad_h, size_t pad_w,
                                           const std::string &name)
    : StatelessLayer(name), pool_h_(pool_h), pool_w_(pool_w),
      stride_h_(stride_h == 0 ? pool_h : stride_h), stride_w_(stride_w == 0 ? pool_w : stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {

  if (pool_h_ == 0 || pool_w_ == 0) {
    throw std::invalid_argument("Pool dimensions must be positive");
  }
  if (stride_h_ == 0 || stride_w_ == 0) {
    throw std::invalid_argument("Stride dimensions must be positive");
  }
}

void LegacyMaxPool2DLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  const auto &shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("MaxPool2D: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t input_h = shape[2];
  const size_t input_w = shape[3];

  micro_batch_input_shapes_[mb_id] = {batch_size, channels, input_h, input_w};

  const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  output->ensure({batch_size, channels, output_h, output_w});

  Tensor &mask_indices = micro_batch_mask_indices_[mb_id];
  if (mask_indices == nullptr)
    mask_indices = Tensor::create<size_t>({batch_size, channels, output_h, output_w});
  else {
    mask_indices->ensure({batch_size, channels, output_h, output_w});
  }

  compute_max_pool_forward(input, output, batch_size, channels, input_h, input_w, output_h,
                           output_w, micro_batch_mask_indices_[mb_id], "default");
}

void LegacyMaxPool2DLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  auto it_mask = micro_batch_mask_indices_.find(mb_id);
  auto it_shape = micro_batch_input_shapes_.find(mb_id);

  if (it_mask == micro_batch_mask_indices_.end()) {
    throw std::runtime_error("No cached mask found for micro-batch ID in LegacyMaxPool2DLayer: " +
                             std::to_string(mb_id));
  }
  if (it_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error(
        "No cached input shape found for micro-batch ID in LegacyMaxPool2DLayer: " +
        std::to_string(mb_id));
  }

  const Tensor &mask_indices = it_mask->second;
  const std::vector<size_t> &input_shape = it_shape->second;

  const size_t batch_size = input_shape[0];
  const size_t channels = input_shape[1];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const auto &grad_shape = gradient->shape();
  if (grad_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2D: Gradient tensor must be 4-dimensional (NCHW)");
  }
  const size_t output_h = grad_shape[2];
  const size_t output_w = grad_shape[3];

  grad_input->ensure({batch_size, channels, input_h, input_w});

  grad_input->fill(0);

  compute_max_pool_backward(gradient, grad_input, batch_size, channels, output_h, output_w,
                            mask_indices, "default");
}

template <typename IO_T>
std::unique_ptr<Task> LegacyMaxPool2DLayer::compute_max_pool_forward_impl(
    const Tensor &input_data, Tensor &output_data, size_t batch_size, size_t channels,
    size_t input_h, size_t input_w, size_t output_h, size_t output_w, Tensor &mask_indices,
    const std::string &flow_id) const {
  if (input_data->data_type() != dtype_of<IO_T>() || output_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyMaxPool2DLayer tensor dtype mismatch with dispatch type");
  }
  if (input_data->device_type() != output_data->device_type()) {
    throw std::runtime_error("Input and output tensors must be on the same device");
  }

  if (input_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::maxpool_nchw::compute_max_pool_forward<IO_T>,
                           input_data->data_as<IO_T>(), output_data->data_as<IO_T>(), batch_size,
                           channels, input_h, input_w, output_h, output_w, pool_h_, pool_w_,
                           stride_h_, stride_w_, pad_h_, pad_w_, mask_indices->data_as<size_t>());
  }
#ifdef USE_CUDA
  else if (input_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::maxpool_nchw::compute_max_pool_forward<IO_T>,
                            input_data->data_as<IO_T>(), output_data->data_as<IO_T>(), batch_size,
                            channels, input_h, input_w, output_h, output_w, pool_h_, pool_w_,
                            stride_h_, stride_w_, pad_h_, pad_w_, mask_indices->data_as<size_t>());
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_max_pool_forward");
  }
  return nullptr;
}

std::unique_ptr<Task> LegacyMaxPool2DLayer::compute_max_pool_forward(
    const Tensor &input_data, Tensor &output_data, size_t batch_size, size_t channels,
    size_t input_h, size_t input_w, size_t output_h, size_t output_w, Tensor &mask_indices,
    const std::string &flow_id) const {
  DISPATCH_ON_DTYPE_TO_METHOD(compute_max_pool_forward_impl, input_data, output_data, batch_size,
                              channels, input_h, input_w, output_h, output_w, mask_indices,
                              flow_id);
  return nullptr;
}

template <typename IO_T>
std::unique_ptr<Task> LegacyMaxPool2DLayer::compute_max_pool_backward_impl(
    const Tensor &gradient_data, Tensor &grad_input_data, size_t batch_size, size_t channels,
    size_t output_h, size_t output_w, const Tensor &mask_indices,
    const std::string &flow_id) const {
  if (gradient_data->data_type() != dtype_of<IO_T>() ||
      grad_input_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyMaxPool2DLayer tensor dtype mismatch with dispatch type");
  }
  if (gradient_data->device_type() != grad_input_data->device_type()) {
    throw std::runtime_error("Gradient and input gradient tensors must be on the same device");
  }

  if (gradient_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::maxpool_nchw::compute_max_pool_backward<IO_T>,
                           gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
                           batch_size, channels, output_h, output_w,
                           mask_indices->data_as<size_t>());
  }
#ifdef USE_CUDA
  else if (gradient_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::maxpool_nchw::compute_max_pool_backward<IO_T>,
                            gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
                            batch_size, channels, output_h, output_w,
                            mask_indices->data_as<size_t>());
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_max_pool_backward");
  }
  return nullptr;
}

std::unique_ptr<Task> LegacyMaxPool2DLayer::compute_max_pool_backward(
    const Tensor &gradient_data, Tensor &grad_input_data, size_t batch_size, size_t channels,
    size_t output_h, size_t output_w, const Tensor &mask_indices,
    const std::string &flow_id) const {
  DISPATCH_ON_DTYPE_TO_METHOD(compute_max_pool_backward_impl, gradient_data, grad_input_data,
                              batch_size, channels, output_h, output_w, mask_indices, flow_id);
  return nullptr;
}

LayerConfig LegacyMaxPool2DLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["pool_h"] = pool_h_;
  config.parameters["pool_w"] = pool_w_;
  config.parameters["stride_h"] = stride_h_;
  config.parameters["stride_w"] = stride_w_;
  config.parameters["pad_h"] = pad_h_;
  config.parameters["pad_w"] = pad_w_;
  return config;
}

std::unique_ptr<Layer> LegacyMaxPool2DLayer::clone() const {
  return std::make_unique<LegacyMaxPool2DLayer>(pool_h_, pool_w_, stride_h_, stride_w_, pad_h_,
                                                pad_w_, this->name_);
}

std::vector<size_t>
LegacyMaxPool2DLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("LegacyMaxPool2DLayer expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t output_h = (input_shape[2] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  return {batch_size, channels, output_h, output_w};
}

std::unique_ptr<LegacyMaxPool2DLayer>
LegacyMaxPool2DLayer::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<LegacyMaxPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                                config.name);
}

uint64_t LegacyMaxPool2DLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  uint64_t comparisons_per_output = pool_h_ * pool_w_;
  uint64_t total_outputs = batch_size * channels * output_h * output_w;

  return comparisons_per_output * total_outputs;
}

uint64_t LegacyMaxPool2DLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  return batch_size * channels * output_h * output_w;
}

} // namespace tnn
