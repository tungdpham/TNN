/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/elu_layer.hpp"

#include <memory>
#include <stdexcept>

namespace synet {

ELULayerImpl::ELULayerImpl(float alpha, const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<ELU>(alpha)),
      alpha_(alpha) {}

Tensor ELULayerImpl::forward_impl(const ConstTensor &input, size_t mb_id) {
  if (this->is_training_) {
    // Cache input for backward pass (ELU gradient requires input values)
    set_immutable_cache(mb_id, "input", input);
  }

  Tensor output = get_tensor(input->shape(), io_dtype_);
  activation_->apply(input, output);
  return output;
}

Tensor ELULayerImpl::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  const ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in ELULayerImpl");
  }

  Tensor grad_input = get_tensor(input->shape(), io_dtype_);
  activation_->compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig ELULayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("alpha", alpha_);
  return config;
}

std::shared_ptr<ELULayerImpl> ELULayerImpl::create_from_config(const LayerConfig &config) {
  float alpha = config.get<float>("alpha", 1.0f);
  return std::make_shared<ELULayerImpl>(alpha, config.name);
}

}  // namespace synet
