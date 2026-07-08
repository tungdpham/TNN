/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/avgpool2d_layer.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

AvgPool2DLayerImpl::AvgPool2DLayerImpl(size_t pool_h, size_t pool_w, size_t stride_h,
                                       size_t stride_w, size_t pad_h, size_t pad_w,
                                       const std::string &name)
    : SISOLayerImpl(name),
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

Tensor AvgPool2DLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 4) {
    throw std::runtime_error("AvgPool2DLayerImpl: input must be 4D (NHWC format)");
  }

  const auto &shape = input.shape();
  size_t batch_size = shape[0];
  size_t input_h = shape[1];
  size_t input_w = shape[2];
  size_t channels = shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  AvgPool2DStats stats{.batch_size = batch_size,
                       .height = input_h,
                       .width = input_w,
                       .channels = channels,
                       .pool_h = pool_h_,
                       .pool_w = pool_w_,
                       .stride_h = stride_h_,
                       .stride_w = stride_w_,
                       .pad_h = pad_h_,
                       .pad_w = pad_w_};

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_avgpool_graph(backend_handle_, stats, type_desc);

  Tensor output = get_tensor({batch_size, output_h, output_w, channels}, input.dtype());
  Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->avgpool_fwd(backend_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                       ws.data_as<void>(), type_desc);

  return output;
}

Tensor AvgPool2DLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.dims() != 4) {
    throw std::runtime_error("AvgPool2DLayerImpl: grad_output must be 4D (NHWC format)");
  }

  const auto &grad_shape = grad_output.shape();
  size_t batch_size = grad_shape[0];
  size_t output_h = grad_shape[1];
  size_t output_w = grad_shape[2];
  size_t channels = grad_shape[3];
  size_t input_h = (output_h - 1) * stride_h_ + pool_h_ - 2 * pad_h_;
  size_t input_w = (output_w - 1) * stride_w_ + pool_w_ - 2 * pad_w_;

  AvgPool2DStats stats{.batch_size = batch_size,
                       .height = input_h,
                       .width = input_w,
                       .channels = channels,
                       .pool_h = pool_h_,
                       .pool_w = pool_w_,
                       .stride_h = stride_h_,
                       .stride_w = stride_w_,
                       .pad_h = pad_h_,
                       .pad_w = pad_w_};

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  Tensor grad_input = get_tensor({batch_size, input_h, input_w, channels}, grad_output.dtype());
  fill(grad_input, 0.0f);

  WorkspaceReq ws_req = engine_->query_avgpool_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->avgpool_bwd(backend_handle_, stats, grad_output.data_as<void>(),
                       grad_input.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig AvgPool2DLayerImpl::get_config() const {
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

Vec<size_t> AvgPool2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("AvgPool2DLayerImpl: input shape must be 4D (NHWC format)");
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

std::shared_ptr<AvgPool2DLayerImpl> AvgPool2DLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_shared<AvgPool2DLayerImpl>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                              config.name);
}

}  // namespace tunx
