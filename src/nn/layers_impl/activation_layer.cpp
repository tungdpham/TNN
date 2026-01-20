/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/activation_layer.hpp"
namespace tnn {

ActivationLayer::ActivationLayer(std::unique_ptr<ActivationFunction> activation,
                                 const std::string &name)
    : StatelessLayer(name), activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Function function cannot be null");
  }
}

void ActivationLayer::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  if (this->is_training_) {
    // Tensor &cached_input = micro_batch_inputs_[micro_batch_id];
    // if (!cached_input) {
    //   cached_input = make_tensor<float>(input->shape(), this->device_);
    // }
    // cached_input->ensure(input->shape(), this->device_);
    // input->copy_to(cached_input);
    micro_batch_inputs_[micro_batch_id] = input;
  }

  output->ensure(input->shape(), this->device_);
  activation_->apply(input, output);
}

void ActivationLayer::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                    size_t micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  if (it_input == micro_batch_inputs_.end()) {
    throw std::runtime_error("Input for micro batch not found");
  }
  const Tensor &input = it_input->second;
  grad_input->ensure(input->shape(), this->device_);
  activation_->compute_gradient(input, gradient, grad_input);
}

std::string ActivationLayer::type() const { return "activation"; }

LayerConfig ActivationLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["activation"] = activation_->name();
  return config;
}

std::unique_ptr<Layer> ActivationLayer::clone() const {
  return std::make_unique<ActivationLayer>(activation_->clone(), this->name_);
}

std::vector<size_t>
ActivationLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
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

} // namespace tnn
