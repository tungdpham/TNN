/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/gelu_layer.hpp"

#include <memory>
#include <stdexcept>

namespace tunx {

GELULayerImpl::GELULayerImpl(const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<GELU>()) {}

Tensor GELULayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

  Tensor output = get_tensor(input.shape(), io_dtype_);
  activation_->apply(input, output);
  return output;
}

Tensor GELULayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in GELULayerImpl");
  }

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);
  activation_->compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig GELULayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<GELULayerImpl> GELULayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<GELULayerImpl>(config.name);
}

}  // namespace tunx
