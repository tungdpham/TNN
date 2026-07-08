/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/maxpool2d_layer.hpp"

#include <cstddef>
#include <stdexcept>

#include "tensor/tensor_ops.hpp"

namespace tunx {

MaxPool2DLayerImpl::MaxPool2DLayerImpl(size_t pool_h, size_t pool_w, size_t stride_h,
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

MaxPool2DLayerImpl::~MaxPool2DLayerImpl() {}

Tensor MaxPool2DLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::runtime_error("MaxPool2DLayerImpl: input must be 4D (NHWC format)");
  }
  size_t batch_size = shape[0];
  size_t input_h = shape[1];
  size_t input_w = shape[2];
  size_t channels = shape[3];

  size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  MaxPool2DStats stats{.batch_size = batch_size,
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

  WorkspaceReq ws_req = engine_->query_maxpool2d_graph(backend_handle_, stats, type_desc);

  Tensor output = get_tensor({batch_size, output_h, output_w, channels}, input.dtype());
  Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  if (is_training_) {
    Tensor mask_indices =
        this->get_tensor({batch_size, output_h, output_w, channels}, DType_t::INT32);
    residuals["mask_indices"] = mask_indices;

    engine_->maxpool2d_fwd(backend_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                           mask_indices.data_as<void>(), ws.data_as<void>(), type_desc);
  } else {
    engine_->maxpool2d_infer(backend_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                             ws.data_as<void>(), type_desc);
  }

  return output;
}

Tensor MaxPool2DLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &mask_indices = residuals["mask_indices"];

  const auto &grad_shape = grad_output.shape();
  if (grad_shape.size() != 4) {
    throw std::runtime_error("MaxPool2DLayerImpl: grad_output must be 4D (NHWC format)");
  }
  size_t batch_size = grad_shape[0];
  size_t output_h = grad_shape[1];
  size_t output_w = grad_shape[2];
  size_t channels = grad_shape[3];
  size_t input_h = (output_h - 1) * stride_h_ - 2 * pad_h_ + pool_h_;
  size_t input_w = (output_w - 1) * stride_w_ - 2 * pad_w_ + pool_w_;

  MaxPool2DStats stats{.batch_size = batch_size,
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

  WorkspaceReq ws_req = engine_->query_maxpool2d_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->maxpool2d_bwd(backend_handle_, stats, grad_output.data_as<void>(),
                         grad_input.data_as<void>(), mask_indices.data_as<void>(),
                         ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig MaxPool2DLayerImpl::get_config() const {
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

Vec<size_t> MaxPool2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2DLayerImpl: input shape must be 4D (NHWC format)");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;
  size_t channels = input_shape[3];

  return {batch_size, output_h, output_w, channels};
}

std::shared_ptr<MaxPool2DLayerImpl> MaxPool2DLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_shared<MaxPool2DLayerImpl>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                              config.name);
}

}  // namespace tunx
