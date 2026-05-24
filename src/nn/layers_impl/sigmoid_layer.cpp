/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sigmoid_layer.hpp"

#include <memory>
#include <stdexcept>

namespace tnn {

SigmoidLayer::SigmoidLayer(const std::string &name)
    : StatelessLayer(name),
      activation_(std::make_unique<Sigmoid>()) {}

Tensor SigmoidLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  Tensor output = get_output_tensor(input->shape());
  activation_->apply(input, output);

  if (this->is_training_) {
    // Cache output for efficient backward pass
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    set_immutable_cache(mb_id, "output", output);
  }

  return output;
}

Tensor SigmoidLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  const ConstTensor &output = this->get_immutable_cache(mb_id, "output");
  if (!output) {
    throw std::runtime_error("No cached output found for backward pass in SigmoidLayer");
  }

  Tensor grad_input = get_output_tensor(grad_output->shape());

  // Gradient: grad_input = grad_output * output * (1 - output)
  const size_t num_elements = grad_output->size();
  if (grad_output->device_type() == DeviceType::CPU) {
    const float *grad_out_data = grad_output->data_as<float>();
    const float *output_data = output->data_as<float>();
    float *grad_in_data = grad_input->data_as<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      float sig = output_data[i];
      grad_in_data[i] = grad_out_data[i] * sig * (1.0f - sig);
    }
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    throw std::runtime_error("SigmoidLayer: GPU backward not yet implemented");
  }
#endif

  return grad_input;
}

LayerConfig SigmoidLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::unique_ptr<SigmoidLayer> SigmoidLayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<SigmoidLayer>(config.name);
}

}  // namespace tnn
