/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_conv2d_layer.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

LegacyConv2DLayerImpl::LegacyConv2DLayerImpl(size_t in_channels, size_t out_channels,
                                             size_t kernel_h, size_t kernel_w, size_t stride_h,
                                             size_t stride_w, size_t pad_h, size_t pad_w,
                                             bool use_bias, const std::string &name)
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

LegacyConv2DLayerImpl::~LegacyConv2DLayerImpl() {}

void LegacyConv2DLayerImpl::init_impl() {
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
 * ! Support both CPU and CUDA devices but only NCHW data format.
 * @tparam T I/O data type
 * @param input input tensor in NCHW format
 * @param output input tensor in NCHW format
 * @param residuals micro batch id for caching input
 */

Tensor LegacyConv2DLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = input.dim(1);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in LegacyConv2DLayerImpl");
  }

  return def_forward(input, residuals);
}

Tensor LegacyConv2DLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = grad_output.dim(1);

  if (channels != out_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in LegacyConv2DLayerImpl");
  }

  return def_backward(grad_output, residuals);
}

Tensor LegacyConv2DLayerImpl::def_forward(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }
  size_t batch_size = input.dim(0);
  size_t input_h = input.dim(2);
  size_t input_w = input.dim(3);

  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, out_channels_, output_h, output_w}, input.dtype());

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_matrix_size = kernel_size * output_size;

  Tensor col_buffer = get_tensor({col_matrix_size}, io_dtype_);
  residuals["col_buffer"] = col_buffer;

  size_t output_buffer_size = out_channels_ * output_size;
  Tensor temp_output_buffer = get_tensor({output_buffer_size}, io_dtype_);

  im2col(input, col_buffer, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
         this->flow_handle_);

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_conv2d_fwd(backend_handle_, col_buffer.data_as<void>(), weights_.data_as<void>(),
                             temp_output_buffer.data_as<void>(), output_size, kernel_size,
                             out_channels_, type_desc);

  cnhw_to_nchw(temp_output_buffer, output, batch_size, out_channels_, output_h, output_w,
               this->flow_handle_);

  if (use_bias_) {
    engine_->legacy_conv2d_add_bias(backend_handle_, output.data_as<void>(), bias_.data_as<void>(),
                                    batch_size, output_h, output_w, out_channels_, type_desc);
  }

  return output;
}

Tensor LegacyConv2DLayerImpl::def_backward(const Tensor &grad_output, Residuals &residuals) {
  const Vec<size_t> &output_shape = grad_output.shape();
  size_t batch_size = output_shape[0];
  size_t channels = output_shape[1];
  size_t output_h = output_shape[2];
  size_t output_w = output_shape[3];
  size_t input_h = (output_h - 1) * stride_h_ + kernel_h_ - 2 * pad_h_;
  size_t input_w = (output_w - 1) * stride_w_ + kernel_w_ - 2 * pad_w_;

  Tensor grad_input = get_tensor({batch_size, channels, input_h, input_w}, io_dtype_);
  fill(grad_input, 0.0f);  // col2im accumulates, so we need to zero first

  Tensor col_buffer = residuals["col_buffer"];

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_grad_matrix_size = kernel_size * output_size;

  size_t gradient_buffer_size = out_channels_ * output_size;
  Tensor temp_gradient_buffer = get_tensor({gradient_buffer_size}, io_dtype_);
  Tensor temp_col_grad_matrix_buffer = get_tensor({col_grad_matrix_size}, io_dtype_);

  nchw_to_cnhw(grad_output, temp_gradient_buffer, batch_size, out_channels_, output_h, output_w,
               this->flow_handle_);

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_conv2d_wgrad(backend_handle_, col_buffer.data_as<void>(),
                               temp_gradient_buffer.data_as<void>(), grad_weights_.data_as<void>(),
                               output_size, kernel_size, out_channels_, type_desc);

  engine_->legacy_conv2d_dgrad(backend_handle_, temp_gradient_buffer.data_as<void>(),
                               weights_.data_as<void>(), temp_col_grad_matrix_buffer.data_as<void>(),
                               output_size, kernel_size, out_channels_, type_desc);

  col2im(temp_col_grad_matrix_buffer, grad_input, batch_size, in_channels_, input_h, input_w,
         kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, this->flow_handle_);

  if (use_bias_) {
    engine_->legacy_conv2d_bgrad(backend_handle_, grad_output.data_as<void>(),
                                 grad_bias_.data_as<void>(), batch_size, output_h, output_w,
                                 out_channels_, type_desc);
  }

  return grad_input;
}



LayerConfig LegacyConv2DLayerImpl::get_config() const {
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

Vec<size_t> LegacyConv2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("LegacyConv2DLayerImpl expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, out_channels_, output_h, output_w};
}

std::shared_ptr<LegacyConv2DLayerImpl> LegacyConv2DLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t in_channels = config.get<size_t>("in_channels");
  size_t out_channels = config.get<size_t>("out_channels");
  size_t kernel_h = config.get<size_t>("kernel_h");
  size_t kernel_w = config.get<size_t>("kernel_w");
  size_t stride_h = config.get<size_t>("stride_h", 1);
  size_t stride_w = config.get<size_t>("stride_w", 1);
  size_t pad_h = config.get<size_t>("pad_h", 0);
  size_t pad_w = config.get<size_t>("pad_w", 0);
  bool use_bias = config.get<bool>("use_bias", true);
  return std::make_shared<LegacyConv2DLayerImpl>(in_channels, out_channels, kernel_h, kernel_w,
                                                 stride_h, stride_w, pad_h, pad_w, use_bias,
                                                 config.name);
}

}  // namespace tunx
