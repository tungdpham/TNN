/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/conv2d_layer.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

Conv2DLayerImpl::Conv2DLayerImpl(size_t in_channels, size_t out_channels, size_t kernel_h,
                                 size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                 size_t pad_w, bool use_bias, const std::string &name)
    : SISOLayerImpl(name),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w),
      use_bias_(use_bias) {}

void Conv2DLayerImpl::init_impl() {
  float stddev = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(in_channels_ * kernel_h_ * kernel_w_)));

  long long seed = this->use_seed_ ? this->srand_seed_
                                   : std::chrono::system_clock::now().time_since_epoch().count();

  fill_normal(weights_, 0, stddev, seed);

  if (use_bias_) {
    fill_normal(bias_, 0, stddev, seed);
  }

  fill(grad_weights_, 0.0f);
  if (use_bias_) {
    fill(grad_bias_, 0.0f);
  }
}

/**
 * @brief Perform convolution 2d forward on input and save it to output.
 * ! Only support CUDA device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param input input tensor in NHWC format
 * @param output output tensor in NHWC format
 * @param residuals micro batch id for caching input
 */

Tensor Conv2DLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = input.dim(3);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayerImpl");
  }

  if (this->is_training_) {
    residuals["input"] = input;
  }

  size_t batch_size = input.dim(0);
  size_t input_h = input.dim(1);
  size_t input_w = input.dim(2);

  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, output_h, output_w, out_channels_}, input.dtype());

  Conv2DStats stats{
      .batch_size = batch_size,
      .in_channels = in_channels_,
      .out_channels = out_channels_,
      .input_h = input_h,
      .input_w = input_w,
      .kernel_h = kernel_h_,
      .kernel_w = kernel_w_,
      .stride_h = stride_h_,
      .stride_w = stride_w_,
      .pad_h = pad_h_,
      .pad_w = pad_w_,
      .use_bias = use_bias_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_conv2d_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->conv2d_fwd(backend_handle_, stats, input.data_as<void>(), weights_.data_as<void>(),
                      use_bias_ ? bias_.data_as<void>() : nullptr, output.data_as<void>(),
                      ws.data_as<void>(), type_desc);

  return output;
}

Tensor Conv2DLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = grad_output.dim(3);

  if (channels != out_channels_) {
    std::cerr << "Gradient shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in Conv2DLayerImpl");
  }

  const Tensor &input = residuals["input"];
  const auto &input_shape = input.shape();
  Tensor grad_input = get_tensor(input_shape, io_dtype_);

  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];

  Conv2DStats stats{
      .batch_size = batch_size,
      .in_channels = in_channels_,
      .out_channels = out_channels_,
      .input_h = input_h,
      .input_w = input_w,
      .kernel_h = kernel_h_,
      .kernel_w = kernel_w_,
      .stride_h = stride_h_,
      .stride_w = stride_w_,
      .pad_h = pad_h_,
      .pad_w = pad_w_,
      .use_bias = use_bias_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_conv2d_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->conv2d_wgrad(backend_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                        grad_weights_.data_as<void>(), ws.data_as<void>(), type_desc);

  if (use_bias_) {
    engine_->conv2d_bgrad(backend_handle_, stats, grad_output.data_as<void>(),
                          grad_bias_.data_as<void>(), ws.data_as<void>(), type_desc);
  }

  engine_->conv2d_dgrad(backend_handle_, stats, grad_output.data_as<void>(),
                        weights_.data_as<void>(), grad_input.data_as<void>(), ws.data_as<void>(),
                        type_desc);

  return grad_input;
}

LayerConfig Conv2DLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("in_channels", in_channels_);
  config.set("out_channels", out_channels_);
  config.set("kernel_h", kernel_h_);
  config.set("kernel_w", kernel_w_);
  config.set("stride_h", stride_h_);
  config.set("stride_w", stride_w_);
  config.set("pad_h", pad_h_);
  config.set("pad_w", pad_w_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> Conv2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayerImpl expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, output_h, output_w, out_channels_};
}

std::shared_ptr<Conv2DLayerImpl> Conv2DLayerImpl::create_from_config(const LayerConfig &config) {
  size_t in_channels = config.get<size_t>("in_channels");
  size_t out_channels = config.get<size_t>("out_channels");
  size_t kernel_h = config.get<size_t>("kernel_h");
  size_t kernel_w = config.get<size_t>("kernel_w");
  size_t stride_h = config.get<size_t>("stride_h", 1);
  size_t stride_w = config.get<size_t>("stride_w", 1);
  size_t pad_h = config.get<size_t>("pad_h", 0);
  size_t pad_w = config.get<size_t>("pad_w", 0);
  bool use_bias = config.get<bool>("use_bias", true);
  return std::make_shared<Conv2DLayerImpl>(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, use_bias, config.name);
}

}  // namespace tunx
