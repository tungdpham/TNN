/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/linear_layer.hpp"

#include <memory>

namespace tnn {

LinearLayerImpl::LinearLayerImpl(const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<Linear>()) {}

Tensor LinearLayerImpl::forward_impl(const ConstTensor &input, size_t mb_id) {
  // Linear activation is identity, just copy the input
  Tensor output = get_output_tensor(input->shape());
  output->share_from(input);
  return output;
}

Tensor LinearLayerImpl::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  // Gradient of identity is identity, just copy grad_output
  Tensor grad_input = get_output_tensor(grad_output->shape());
  grad_input->share_from(grad_output);
  return grad_input;
}

LayerConfig LinearLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::unique_ptr<LinearLayerImpl> LinearLayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_unique<LinearLayerImpl>(config.name);
}

}  // namespace tnn
