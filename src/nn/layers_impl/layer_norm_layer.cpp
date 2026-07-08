/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/layer_norm_layer.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

LayerNormLayerImpl::LayerNormLayerImpl(size_t normalized_shape, float epsilon, bool affine,
                                       const std::string &name)
    : SISOLayerImpl(name),
      normalized_shape_(normalized_shape),
      epsilon_(epsilon),
      affine_(affine) {}

LayerNormLayerImpl::~LayerNormLayerImpl() {}

void LayerNormLayerImpl::init_impl() {
  fill(gamma_, 1.0f);
  fill(beta_, 0.0f);

  fill(grad_gamma_, 0.0f);
  fill(grad_beta_, 0.0f);
}

Tensor LayerNormLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

  const auto &shape = input.shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  LayerNormStats stats{
      .batch_size = batch_size,
      .seq_len = 1,
      .channels = channels,
      .epsilon = epsilon_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_layernorm_graph(backend_handle_, stats, type_desc);

  if (this->is_training_) {
    Tensor batch_mean = get_tensor({batch_size}, compute_dtype_);
    Tensor batch_invar = get_tensor({batch_size}, compute_dtype_);
    residuals["batch_mean"] = batch_mean;
    residuals["batch_invar"] = batch_invar;

    Tensor output = get_tensor(shape, io_dtype_);

    Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

    engine_->layernorm_fwd(backend_handle_, stats, input.data_as<void>(), gamma_.data_as<void>(),
                           beta_.data_as<void>(), output.data_as<void>(),
                           batch_mean.data_as<void>(), batch_invar.data_as<void>(),
                           ws.data_as<void>(), type_desc);

    return output;
  } else {
    Tensor output = get_tensor(shape, io_dtype_);

    Tensor ws = get_tensor({ws_req.inf_workspace}, DType_t::BYTE);

    engine_->layernorm_infer(backend_handle_, stats, input.data_as<void>(), gamma_.data_as<void>(),
                             beta_.data_as<void>(), output.data_as<void>(), ws.data_as<void>(),
                             type_desc);
    return output;
  }
}

Tensor LayerNormLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input.shape();
  Tensor grad_input = get_tensor(shape, io_dtype_);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  LayerNormStats stats{
      .batch_size = batch_size,
      .seq_len = 1,
      .channels = channels,
      .epsilon = epsilon_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_layernorm_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  const Tensor &batch_mean = residuals["batch_mean"];
  const Tensor &batch_invar = residuals["batch_invar"];

  if (affine_) {
    engine_->layernorm_bwd(backend_handle_, stats, grad_output.data_as<void>(),
                           input.data_as<void>(), gamma_.data_as<void>(),
                           batch_mean.data_as<void>(), batch_invar.data_as<void>(),
                           grad_input.data_as<void>(), grad_gamma_.data_as<void>(),
                           grad_beta_.data_as<void>(), ws.data_as<void>(), type_desc);
  } else {
    engine_->layernorm_bwd(
        backend_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
        gamma_.data_as<void>(), batch_mean.data_as<void>(), batch_invar.data_as<void>(),
        grad_input.data_as<void>(), nullptr, nullptr, ws.data_as<void>(), type_desc);
  }

  return grad_input;
}

LayerConfig LayerNormLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("normalized_shape", normalized_shape_);
  config.set("epsilon", epsilon_);
  config.set("affine", affine_);
  return config;
}

std::shared_ptr<LayerNormLayerImpl> LayerNormLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t normalized_shape = config.get<size_t>("normalized_shape");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine", true);
  return std::make_shared<LayerNormLayerImpl>(normalized_shape, epsilon, affine, config.name);
}

}  // namespace tunx
