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
    : StatelessLayer(name), activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Function function cannot be null");
  }
}

void ActivationLayer::forward_impl(const ConstTensor &input, Tensor &output, size_t mb_id) {
  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  output->ensure(input->shape());
  activation_->apply(input, output);
}

void ActivationLayer::backward_impl(const ConstTensor &gradient, Tensor &grad_input, size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in ActivationLayer");
  }
  grad_input->ensure(input->shape());
  activation_->compute_gradient(input, gradient, grad_input);
}

LayerConfig ActivationLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["activation"] = activation_->name();
  return config;
}

std::unique_ptr<Layer> ActivationLayer::clone() const {
  return std::make_unique<ActivationLayer>(activation_->clone(), this->name_);
}

std::vector<size_t> ActivationLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  return input_shape;
}

uint64_t ActivationLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return 2 * num_elements;
}

uint64_t ActivationLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return 2 * num_elements;
}

std::unique_ptr<ActivationLayer> ActivationLayer::create_from_config(const LayerConfig &config) {
  std::string activation_name = config.get<std::string>("activation", "relu");
  ActivationFactory::register_defaults();
  auto activation = ActivationFactory::create(activation_name);
  return std::make_unique<ActivationLayer>(std::move(activation), config.name);
}

}  // namespace tnn
