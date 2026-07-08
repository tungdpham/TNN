/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_avgpool2d_layer.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

namespace tunx {

LegacyAvgPool2DLayerImpl::LegacyAvgPool2DLayerImpl(size_t pool_h, size_t pool_w, size_t stride_h,
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

Tensor LegacyAvgPool2DLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 4) {
    throw std::invalid_argument("AvgPool2D: Input tensor must be 4-dimensional (NCHW)");
  }

  const auto &shape = input.shape();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t input_h = shape[2];
  size_t input_w = shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, channels, output_h, output_w}, input.dtype());

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_avgpool2d_fwd(backend_handle_, input.data_as<void>(), output.data_as<void>(),
                                batch_size, channels, input_h, input_w, output_h, output_w, pool_h_,
                                pool_w_, stride_h_, stride_w_, pad_h_, pad_w_, type_desc);

  return output;
}

Tensor LegacyAvgPool2DLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("AvgPool2D: Gradient tensor must be 4-dimensional (NCHW)");
  }
  const auto &grad_shape = grad_output.shape();

  if (grad_shape.size() != 4) {
    throw std::invalid_argument("AvgPool2D: Gradient tensor must be 4-dimensional (NCHW)");
  }

  size_t batch_size = grad_shape[0];
  size_t channels = grad_shape[1];
  size_t output_h = grad_shape[2];
  size_t output_w = grad_shape[3];
  size_t input_h = (grad_shape[2] - 1) * stride_h_ + pool_h_ - 2 * pad_h_;
  size_t input_w = (grad_shape[3] - 1) * stride_w_ + pool_w_ - 2 * pad_w_;

  Tensor grad_input = get_tensor({batch_size, channels, input_h, input_w}, grad_output.dtype());
  fill(grad_input, 0.0f);

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_avgpool2d_bwd(backend_handle_, grad_output.data_as<void>(),
                                grad_input.data_as<void>(), batch_size, channels, input_h, input_w,
                                output_h, output_w, pool_h_, pool_w_, stride_h_, stride_w_, pad_h_,
                                pad_w_, type_desc);

  return grad_input;
}



LayerConfig LegacyAvgPool2DLayerImpl::get_config() const {
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

Vec<size_t> LegacyAvgPool2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("LegacyAvgPool2DLayerImpl expects 4D input including batch size");
  }

  // Check for underflow in the calculation
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t padded_h = input_shape[2] + 2 * pad_h_;
  size_t padded_w = input_shape[3] + 2 * pad_w_;

  size_t output_h = (padded_h - pool_h_) / stride_h_ + 1;
  size_t output_w = (padded_w - pool_w_) / stride_w_ + 1;

  return {batch_size, channels, output_h, output_w};
}

std::shared_ptr<LegacyAvgPool2DLayerImpl> LegacyAvgPool2DLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_shared<LegacyAvgPool2DLayerImpl>(pool_h, pool_w, stride_h, stride_w, pad_h,
                                                    pad_w, config.name);
}

}  // namespace tunx
