/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dropout_layer.hpp"

#include <memory>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "type/type.hpp"

namespace tunx {

DropoutLayerImpl::DropoutLayerImpl(float dropout_rate, const std::string &name)
    : SISOLayerImpl(name),
      dropout_rate_(dropout_rate),
      generator_(std::random_device{}()) {
  if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
    throw std::invalid_argument("Dropout rate must be in [0, 1)");
  }
}

Tensor DropoutLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (!this->is_training_) {
    Tensor output = get_tensor(input.shape(), io_dtype_);
    input.copy_to(output);
    return output;
  }

  size_t batch_size = input.dim(0);
  size_t channels = input.dims() > 1 ? input.dim(1) : 1;
  size_t spatial_size = input.dims() > 1 ? input.stride(1) : input.stride(0);

  DropoutStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .spatial_size = spatial_size,
      .dropout_rate = dropout_rate_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_dropout_graph(backend_handle_, stats, type_desc);

  Tensor mask = get_tensor(input.shape(), DType_t::BOOL);
  residuals["mask"] = mask;

  Tensor output = get_tensor(input.shape(), io_dtype_);
  Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->dropout_fwd(backend_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                       mask.data_as<bool>(), ws.data_as<void>(), type_desc);

  return output;
}

Tensor DropoutLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor &mask = residuals["mask"];
  if (!mask) {
    throw std::runtime_error("No cached mask found in DropoutLayerImpl backward pass");
  }

  size_t batch_size = grad_output.dim(0);
  size_t channels = grad_output.dims() > 1 ? grad_output.dim(1) : 1;
  size_t spatial_size = grad_output.dims() > 1 ? grad_output.stride(1) : grad_output.stride(0);

  DropoutStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .spatial_size = spatial_size,
      .dropout_rate = dropout_rate_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  Tensor grad_input = get_tensor(grad_output.shape(), io_dtype_);
  WorkspaceReq ws_req = engine_->query_dropout_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  double scale = 1.0 / (1.0 - dropout_rate_);

  engine_->dropout_bwd(backend_handle_, stats, grad_output.data_as<void>(),
                       grad_input.data_as<void>(), mask.data_as<bool>(), scale, ws.data_as<void>(),
                       type_desc);

  return grad_input;
}

LayerConfig DropoutLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("dropout_rate", dropout_rate_);
  return config;
}

Vec<size_t> DropoutLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<DropoutLayerImpl> DropoutLayerImpl::create_from_config(const LayerConfig &config) {
  float dropout_rate = config.get<float>("dropout_rate");
  return std::make_shared<DropoutLayerImpl>(dropout_rate, config.name);
}

}  // namespace tunx
