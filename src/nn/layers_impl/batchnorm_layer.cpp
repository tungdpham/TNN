/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/batchnorm_layer.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "nn/layer.hpp"
#include "nn/stats/stats.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

BatchNormLayerImpl::BatchNormLayerImpl(size_t num_features, float epsilon, float momentum,
                                       bool affine, bool use_relu, const std::string &name)
    : SISOLayerImpl(name),
      num_features_(num_features),
      epsilon_(epsilon),
      momentum_(momentum),
      affine_(affine),
      use_relu_(use_relu) {}

BatchNormLayerImpl::~BatchNormLayerImpl() {}

void BatchNormLayerImpl::init_impl() {
  fill(gamma_, 1.0f);
  fill(beta_, 0.0f);

  fill(running_mean_, 0.0f);
  fill(running_var_, 1.0f);

  fill(grad_gamma_, 0.0f);
  fill(grad_beta_, 0.0f);

  fill(grad_dummy_mean_, 0.0f);
  fill(grad_dummy_var_, 0.0f);
}

/**
 * @brief Forward pass for BatchNormLayerImpl
 * @param input Tensor in NHWC format
 * @param output Tensor in NHWC format
 * @param residuals Micro-batch identifier for caching
 */
Tensor BatchNormLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() < 4) {
    throw std::invalid_argument("BatchNorm: Input tensor must have at least 4 dimensions got " +
                                std::to_string(input.dims()) + " dims");
  }
  if (input.dim(3) != num_features_) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features" +
                                std::to_string(num_features_) + ", but got " +
                                std::to_string(input.dim(3)));
  }

  size_t N = input.dim(0);
  size_t H = input.dim(1);
  size_t W = input.dim(2);
  size_t C = input.dim(3);

  BatchNormStats stats{
      .batch_size = N,
      .height = H,
      .width = W,
      .channels = C,
      .epsilon = epsilon_,
      .momentum = momentum_,
      .use_relu = use_relu_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_batchnorm_graph(backend_handle_, stats, type_desc);

  // allocate order: residuals, output, workspace.
  if (is_training_) {
    residuals["input"] = input;

    Tensor batch_mean = get_tensor({C}, DType_t::FP32);
    Tensor batch_invar = get_tensor({C}, DType_t::FP32);
    residuals["batch_mean"] = batch_mean;
    residuals["batch_invar"] = batch_invar;

    Tensor relu_mask;
    if (use_relu_) {
      relu_mask = get_tensor(input.shape(), DType_t::BOOL);
      residuals["relu_mask"] = relu_mask;
    }

    Tensor output = get_tensor(input.shape(), io_dtype_);

    Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

    engine_->batchnorm_fwd(
        backend_handle_, stats, input.data_as<void>(), gamma_.data_as<void>(),
        beta_.data_as<void>(), output.data_as<void>(), running_mean_.data_as<void>(),
        running_var_.data_as<void>(), running_mean_.data_as<void>(), running_var_.data_as<void>(),
        batch_mean.data_as<void>(), batch_invar.data_as<void>(),
        use_relu_ ? relu_mask.data_as<void>() : nullptr, ws.data_as<void>(), type_desc);

    return output;
  } else {
    Tensor output = get_tensor(input.shape(), io_dtype_);

    Tensor ws = get_tensor({ws_req.inf_workspace}, DType_t::BYTE);

    engine_->batchnorm_infer(backend_handle_, stats, input.data_as<void>(), gamma_.data_as<void>(),
                             beta_.data_as<void>(), running_mean_.data_as<void>(),
                             running_var_.data_as<void>(), output.data_as<void>(),
                             ws.data_as<void>(), type_desc);

    return output;
  }
}

Tensor BatchNormLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  Tensor &batch_mean = residuals["batch_mean"];
  Tensor &batch_invar = residuals["batch_invar"];
  Tensor relu_mask = use_relu_ ? residuals["relu_mask"] : Tensor();

  size_t N = grad_output.dim(0);
  size_t H = grad_output.dim(1);
  size_t W = grad_output.dim(2);
  size_t C = grad_output.dim(3);

  BatchNormStats stats{
      .batch_size = N,
      .height = H,
      .width = W,
      .channels = C,
      .epsilon = epsilon_,
      .momentum = momentum_,
      .use_relu = use_relu_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  Tensor grad_input = get_tensor(grad_output.shape(), io_dtype_);

  WorkspaceReq ws_req = engine_->query_batchnorm_graph(backend_handle_, stats, type_desc);
  Tensor workspace = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->batchnorm_bwd(backend_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         use_relu_ ? relu_mask.data_as<void>() : nullptr, gamma_.data_as<void>(),
                         grad_input.data_as<void>(), grad_gamma_.data_as<void>(),
                         grad_beta_.data_as<void>(), batch_mean.data_as<void>(),
                         batch_invar.data_as<void>(), workspace.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig BatchNormLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("num_features", num_features_);
  config.set("epsilon", epsilon_);
  config.set("momentum", momentum_);
  config.set("affine", affine_);
  config.set("use_relu", use_relu_);
  return config;
}

Vec<size_t> BatchNormLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<BatchNormLayerImpl> BatchNormLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  float epsilon = config.get<float>("epsilon");
  float momentum = config.get<float>("momentum");
  bool affine = config.get<bool>("affine");
  bool use_relu = config.get<bool>("use_relu", false);

  return std::make_shared<BatchNormLayerImpl>(num_features, epsilon, momentum, affine, use_relu,
                                              config.name);
}

}  // namespace tunx
