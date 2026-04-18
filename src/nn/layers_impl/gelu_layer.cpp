/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/gelu_layer.hpp"

#include <memory>
#include <stdexcept>

namespace tnn {

GELULayer::GELULayer(const std::string &name)
    : StatelessLayer(name),
      activation_(std::make_unique<GELU>()) {}

Tensor GELULayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  if (this->is_training_) {
    // Cache input for backward pass (GELU gradient requires input values)
    set_immutable_cache(mb_id, "input", input);
  }

  Tensor output = get_output_tensor(input->shape());
  activation_->apply(input, output);
  return output;
}

Tensor GELULayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  const ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in GELULayer");
  }

  Tensor grad_input = get_output_tensor(input->shape());
  activation_->compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig GELULayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::unique_ptr<GELULayer> GELULayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<GELULayer>(config.name);
}

}  // namespace tnn
