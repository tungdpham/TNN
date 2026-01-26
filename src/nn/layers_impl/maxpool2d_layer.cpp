/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/maxpool_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/maxpool_ops.hpp"
#endif
#include "tensor/tensor.hpp"

#include <cstddef>
#include <stdexcept>

namespace tnn {

MaxPool2DLayer::MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                               size_t pad_h, size_t pad_w, const std::string &name)
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

void MaxPool2DLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  const auto &shape = input->shape();
  if (shape.size() != 4) {
    throw std::runtime_error("MaxPool2DLayer: input must be 4D (NHWC format)");
  }
  const size_t batch_size = shape[0];
  const size_t input_h = shape[1];
  const size_t input_w = shape[2];
  const size_t channels = shape[3];

  micro_batch_input_shapes_[mb_id] = {batch_size, input_h, input_w, channels};

  const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  output->ensure({batch_size, output_h, output_w, channels});

  Tensor &mask_indices = micro_batch_mask_indices_[mb_id];
  if (mask_indices == nullptr)
    mask_indices = Tensor::create<int>({batch_size, output_h, output_w, channels}, this->device_);
  else {
    mask_indices->ensure({batch_size, output_h, output_w, channels});
  }

  compute_max_pool_forward(input, output, batch_size, input_h, input_w, channels, output_h,
                           output_w, micro_batch_mask_indices_[mb_id], "default");
}

void MaxPool2DLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  auto it_mask = micro_batch_mask_indices_.find(mb_id);
  auto it_shape = micro_batch_input_shapes_.find(mb_id);

  if (it_mask == micro_batch_mask_indices_.end()) {
    throw std::runtime_error("MaxPool2DLayer: forward must be called before backward");
  }
  if (it_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("MaxPool2DLayer: forward must be called before backward");
  }

  const Tensor &mask_indices = it_mask->second;
  const std::vector<size_t> &input_shape = it_shape->second;

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[1];
  const size_t input_w = input_shape[2];
  const size_t channels = input_shape[3];
  const auto &grad_shape = gradient->shape();
  if (grad_shape.size() != 4) {
    throw std::runtime_error("MaxPool2DLayer: gradient must be 4D (NHWC format)");
  }
  const size_t output_h = grad_shape[1];
  const size_t output_w = grad_shape[2];

  grad_input->ensure({batch_size, input_h, input_w, channels});

  grad_input->fill(0);

  compute_max_pool_backward(gradient, grad_input, batch_size, channels, output_h, output_w,
                            mask_indices, "default");
}

template <typename IO_T>
std::unique_ptr<Task> MaxPool2DLayer::compute_max_pool_forward_impl(
    const Tensor &input_data, Tensor &output_data, size_t batch_size, size_t height, size_t width,
    size_t channels, size_t output_h, size_t output_w, Tensor &mask_indices,
    const std::string &flow_id) const {
  if (input_data->data_type() != dtype_of<IO_T>() || output_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("MaxPool2DLayer: data type mismatch in forward pass");
  }

  if (input_data->device_type() == DeviceType::CPU) {
    cpu::maxpool_forward<IO_T>(input_data->data_as<IO_T>(), output_data->data_as<IO_T>(),
                               mask_indices->data_as<int>(), batch_size, height, width, channels,
                               pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_, output_h,
                               output_w);
  }
#ifdef USE_CUDA
  else if (input_data->device_type() == DeviceType::GPU) {
    cuda::maxpool_forward<IO_T>(input_data->data_as<IO_T>(), output_data->data_as<IO_T>(),
                                mask_indices->data_as<int>(), batch_size, height, width, channels,
                                pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_, output_h,
                                output_w);
  }
#endif
  else {
    throw std::runtime_error("MaxPool2DLayer: unsupported device type");
  }
  return nullptr;
}

std::unique_ptr<Task>
MaxPool2DLayer::compute_max_pool_forward(const Tensor &input_data, Tensor &output_data,
                                         size_t batch_size, size_t height, size_t width,
                                         size_t channels, size_t output_h, size_t output_w,
                                         Tensor &mask_indices, const std::string &flow_id) const {
  DISPATCH_ON_DTYPE_TO_METHOD(compute_max_pool_forward_impl, input_data, output_data, batch_size,
                              height, width, channels, output_h, output_w, mask_indices, flow_id);
  return nullptr;
}

template <typename IO_T>
std::unique_ptr<Task>
MaxPool2DLayer::compute_max_pool_backward_impl(const Tensor &gradient_data, Tensor &grad_input_data,
                                               size_t batch_size, size_t channels, size_t output_h,
                                               size_t output_w, const Tensor &mask_indices,
                                               const std::string &flow_id) const {
  if (gradient_data->data_type() != dtype_of<IO_T>() ||
      grad_input_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("MaxPool2DLayer: data type mismatch in backward pass");
  }

  if (gradient_data->device_type() == DeviceType::CPU) {
    cpu::maxpool_backward<IO_T>(gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
                                mask_indices->data_as<int>(), batch_size, channels, output_h,
                                output_w);
  }
#ifdef USE_CUDA
  else if (gradient_data->device_type() == DeviceType::GPU) {
    cuda::maxpool_backward<IO_T>(gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
                                 mask_indices->data_as<int>(), batch_size, channels, output_h,
                                 output_w);
  }
#endif
  else {
    throw std::runtime_error("MaxPool2DLayer: unsupported device type");
  }
  return nullptr;
}

std::unique_ptr<Task> MaxPool2DLayer::compute_max_pool_backward(const Tensor &gradient_data,
                                                                Tensor &grad_input_data,
                                                                size_t batch_size, size_t channels,
                                                                size_t output_h, size_t output_w,
                                                                const Tensor &mask_indices,
                                                                const std::string &flow_id) const {
  DISPATCH_ON_DTYPE_TO_METHOD(compute_max_pool_backward_impl, gradient_data, grad_input_data,
                              batch_size, channels, output_h, output_w, mask_indices, flow_id);
  return nullptr;
}

LayerConfig MaxPool2DLayer::get_config() const {
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

std::unique_ptr<Layer> MaxPool2DLayer::clone() const {
  return std::make_unique<MaxPool2DLayer>(pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                                          this->name_);
}

std::vector<size_t>
MaxPool2DLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2DLayer: input shape must be 4D (NHWC format)");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;
  size_t channels = input_shape[3];

  return {batch_size, output_h, output_w, channels};
}

std::unique_ptr<MaxPool2DLayer> MaxPool2DLayer::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<MaxPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                          config.name);
}

uint64_t MaxPool2DLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
  size_t channels = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  uint64_t comparisons_per_output = pool_h_ * pool_w_;
  uint64_t total_outputs = batch_size * output_h * output_w * channels;

  return comparisons_per_output * total_outputs;
}

uint64_t MaxPool2DLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
  size_t channels = input_shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  return batch_size * output_h * output_w * channels;
}

} // namespace tnn
