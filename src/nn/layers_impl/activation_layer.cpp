/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/activation_layer.hpp"

#include "nn/activations.hpp"
namespace tnn {

ActivationLayer::ActivationLayer(std::unique_ptr<ActivationFunction> activation,
                                 const std::string &name)
    : StatelessLayer(name),
      activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Function function cannot be null");
  }
}

Tensor ActivationLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  if (this->is_training_) {
    set_immutable_cache(mb_id, "input", input);
  }

  Tensor output = get_output_tensor(input->shape());
  activation_->apply(input, output);
  return output;
}

Tensor ActivationLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in ActivationLayer");
  }
  Tensor grad_input = get_output_tensor(input->shape());
  activation_->compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig ActivationLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("activation", activation_->name());
  return config;
}

Vec<size_t> ActivationLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::unique_ptr<ActivationLayer> ActivationLayer::create_from_config(const LayerConfig &config) {
  std::string activation_name = config.get<std::string>("activation", "relu");
  ActivationFactory::register_defaults();
  auto activation = ActivationFactory::create(activation_name);
  return std::make_unique<ActivationLayer>(std::move(activation), config.name);
}

}  // namespace tnn
