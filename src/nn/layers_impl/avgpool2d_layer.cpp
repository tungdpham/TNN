/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/avgpool2d_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/avgpool_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/avgpool_ops.hpp"
#endif
#include <stdexcept>

namespace tnn {

AvgPool2DLayer::AvgPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                               size_t pad_h, size_t pad_w, const std::string &name)
    : StatelessLayer(name),
      pool_h_(pool_h),
      pool_w_(pool_w),
      stride_h_(stride_h == 0 ? pool_h : stride_h),
      stride_w_(stride_w == 0 ? pool_w : stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w) {
  if (pool_h_ == 0 || pool_w_ == 0) {
    throw std::invalid_argument("Pool dimensions must be positive");
  }
  if (stride_h_ == 0 || stride_w_ == 0) {
    throw std::invalid_argument("Stride dimensions must be positive");
  }
}

void AvgPool2DLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (input->dims() != 4) {
    throw std::runtime_error("AvgPool2DLayer: input must be 4D (NHWC format)");
  }

  const auto &shape = input->shape();
  const size_t batch_size = shape[0];
  const size_t input_h = shape[1];
  const size_t input_w = shape[2];
  const size_t channels = shape[3];

  micro_batch_input_shapes_[mb_id] = {batch_size, input_h, input_w, channels};

  const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  output->ensure({batch_size, output_h, output_w, channels});

  DISPATCH_ON_DTYPE_TO_METHOD(compute_avg_pool_forward_impl, input, output, batch_size, input_h,
                              input_w, channels, output_h, output_w, "default");
}

void AvgPool2DLayer::backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                                   size_t mb_id) {
  if (gradient->dims() != 4) {
    throw std::runtime_error("AvgPool2DLayer: gradient must be 4D (NHWC format)");
  }
  auto it_shape = micro_batch_input_shapes_.find(mb_id);

  if (it_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("AvgPool2DLayer: forward must be called before backward");
  }

  const auto &input_shape = it_shape->second;
  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[1];
  const size_t input_w = input_shape[2];
  const size_t channels = input_shape[3];
  const auto &grad_shape = gradient->shape();
  const size_t output_h = grad_shape[1];
  const size_t output_w = grad_shape[2];

  grad_input->ensure({batch_size, input_h, input_w, channels});
  grad_input->fill(0);

  DISPATCH_ON_DTYPE_TO_METHOD(compute_avg_pool_backward_impl, gradient, grad_input, batch_size,
                              input_h, input_w, channels, output_h, output_w, "default");
}

template <typename IO_T>
std::unique_ptr<Task> AvgPool2DLayer::compute_avg_pool_forward_impl(
    const ConstTensor &input_data, const Tensor &output_data, size_t batch_size, size_t height,
    size_t width, size_t channels, size_t output_h, size_t output_w,
    const std::string &flow_id) const {
  if (input_data->data_type() != dtype_of<IO_T>() || output_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("AvgPool2DLayer: data type mismatch in forward pass");
  }

  if (input_data->device_type() == DeviceType::CPU) {
    cpu::avgpool_forward<IO_T>(input_data->data_as<IO_T>(), output_data->data_as<IO_T>(),
                               batch_size, height, width, channels, pool_h_, pool_w_, stride_h_,
                               stride_w_, pad_h_, pad_w_, output_h, output_w);
  }
#ifdef USE_CUDA
  else if (input_data->device_type() == DeviceType::GPU) {
    cuda::avgpool_forward<IO_T>(input_data->data_as<IO_T>(), output_data->data_as<IO_T>(),
                                batch_size, height, width, channels, pool_h_, pool_w_, stride_h_,
                                stride_w_, pad_h_, pad_w_, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("AvgPool2DLayer: unsupported device type");
  }
  return nullptr;
}

template <typename IO_T>
std::unique_ptr<Task> AvgPool2DLayer::compute_avg_pool_backward_impl(
    const ConstTensor &gradient_data, const Tensor &grad_input_data, size_t batch_size,
    size_t input_h, size_t input_w, size_t channels, size_t output_h, size_t output_w,
    const std::string &flow_id) const {
  if (gradient_data->data_type() != dtype_of<IO_T>() ||
      grad_input_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("AvgPool2DLayer: data type mismatch in backward pass");
  }

  if (gradient_data->device_type() == DeviceType::CPU) {
    cpu::avgpool_backward<IO_T>(gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
                                batch_size, input_h, input_w, channels, pool_h_, pool_w_, stride_h_,
                                stride_w_, pad_h_, pad_w_, output_h, output_w);
  }
#ifdef USE_CUDA
  else if (gradient_data->device_type() == DeviceType::GPU) {
    cuda::avgpool_backward<IO_T>(gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
                                 batch_size, input_h, input_w, channels, pool_h_, pool_w_,
                                 stride_h_, stride_w_, pad_h_, pad_w_, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("AvgPool2DLayer: unsupported device type");
  }
  return nullptr;
}

LayerConfig AvgPool2DLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("pool_h", pool_h_);
  config.set("pool_w", pool_w_);
  config.set("stride_h", stride_h_);
  config.set("stride_w", stride_w_);
  config.set("pad_h", pad_h_);
  config.set("pad_w", pad_w_);
  return config;
}

std::unique_ptr<Layer> AvgPool2DLayer::clone() const {
  return std::make_unique<AvgPool2DLayer>(pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                                          this->name_);
}

std::vector<size_t> AvgPool2DLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("AvgPool2DLayer: input shape must be 4D (NHWC format)");
  }

  // Check for underflow in the calculation
  size_t batch_size = input_shape[0];
  size_t padded_h = input_shape[1] + 2 * pad_h_;
  size_t padded_w = input_shape[2] + 2 * pad_w_;
  size_t channels = input_shape[3];

  size_t output_h = (padded_h - pool_h_) / stride_h_ + 1;
  size_t output_w = (padded_w - pool_w_) / stride_w_ + 1;

  return {batch_size, output_h, output_w, channels};
}

std::unique_ptr<AvgPool2DLayer> AvgPool2DLayer::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<AvgPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                          config.name);
}

uint64_t AvgPool2DLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
  size_t channels = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  uint64_t flops_per_output = pool_h_ * pool_w_ + 1;
  uint64_t total_outputs = batch_size * output_h * output_w * channels;

  return flops_per_output * total_outputs;
}

uint64_t AvgPool2DLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
  size_t channels = input_shape[3];

  uint64_t flops_per_element = 2;
  uint64_t total_inputs = batch_size * input_h * input_w * channels;

  return flops_per_element * total_inputs;
}

}  // namespace tnn
