/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_batchnorm_layer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

namespace tunx {

LegacyBatchNormLayerImpl::LegacyBatchNormLayerImpl(size_t num_features, float epsilon,
                                                   float momentum, bool affine,
                                                   const std::string &name)
    : SISOLayerImpl(name),
      num_features_(num_features),
      epsilon_(epsilon),
      momentum_(momentum),
      affine_(affine) {}

void LegacyBatchNormLayerImpl::init_impl() {
  fill(gamma_, 1.0f);
  fill(beta_, 0.0f);

  fill(running_mean_, 0.0f);
  fill(running_var_, 1.0f);

  fill(grad_gamma_, 0.0f);
  fill(grad_beta_, 0.0f);

  fill(grad_dummy_mean_, 0.0f);
  fill(grad_dummy_var_, 0.0f);
}

Tensor LegacyBatchNormLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() < 3) {
    throw std::invalid_argument("BatchNorm: Input tensor must have at least 3 dimensions");
  }
  if (input.dim(1) != num_features_) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features");
  }

  return def_forward(input, residuals);
}

Tensor LegacyBatchNormLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  return def_backward(grad_output, residuals);
}

Tensor LegacyBatchNormLayerImpl::def_forward(const Tensor &input, Residuals &residuals) {
  size_t batch_size, channels, spatial_size;
  batch_size = input.dim(0);
  channels = input.dim(1);
  spatial_size = input.stride(1);

  if (num_features_ != channels) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features.");
  }

  Tensor output = get_tensor(input.shape(), io_dtype_);

  Tensor norm = this->get_tensor(input.shape(), io_dtype_);
  Tensor batch_inv_std = this->get_tensor({num_features_}, io_dtype_);
  Tensor batch_mean = this->get_tensor({num_features_}, io_dtype_);

  residuals["norm"] = norm;
  residuals["inv_std"] = batch_inv_std;
  residuals["mean"] = batch_mean;

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  if (this->is_training_) {
    engine_->legacy_batchnorm_fwd(backend_handle_, input.data_as<void>(), batch_mean.data_as<void>(),
                                  batch_inv_std.data_as<void>(), running_mean_.data_as<void>(),
                                  running_var_.data_as<void>(), gamma_.data_as<void>(),
                                  beta_.data_as<void>(), output.data_as<void>(), norm.data_as<void>(),
                                  batch_size, channels, spatial_size, momentum_, epsilon_, affine_,
                                  type_desc);
  } else {
    engine_->legacy_batchnorm_infer(backend_handle_, input.data_as<void>(),
                                    running_mean_.data_as<void>(), running_var_.data_as<void>(),
                                    gamma_.data_as<void>(), beta_.data_as<void>(),
                                    output.data_as<void>(), batch_size, channels, spatial_size,
                                    epsilon_, affine_, type_desc);
  }

  return output;
}

Tensor LegacyBatchNormLayerImpl::def_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &norm = residuals["norm"];
  const Tensor &inv_std = residuals["inv_std"];

  size_t batch_size = grad_output.dim(0);
  size_t channels = grad_output.dim(1);
  size_t spatial_size = grad_output.stride(1);

  Tensor grad_input = get_tensor(grad_output.shape(), io_dtype_);
  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_batchnorm_bwd(backend_handle_, grad_output.data_as<void>(), norm.data_as<void>(),
                                inv_std.data_as<void>(), gamma_.data_as<void>(),
                                grad_gamma_.data_as<void>(), grad_beta_.data_as<void>(),
                                grad_input.data_as<void>(), batch_size, channels, spatial_size,
                                affine_, type_desc);

  return grad_input;
}



LayerConfig LegacyBatchNormLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("num_features", num_features_);
  config.set("epsilon", epsilon_);
  config.set("momentum", momentum_);
  config.set("affine", affine_);
  return config;
}

Vec<size_t> LegacyBatchNormLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<LegacyBatchNormLayerImpl> LegacyBatchNormLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  float epsilon = config.get<float>("epsilon");
  float momentum = config.get<float>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_shared<LegacyBatchNormLayerImpl>(num_features, epsilon, momentum, affine,
                                                    config.name);
}

}  // namespace tunx
