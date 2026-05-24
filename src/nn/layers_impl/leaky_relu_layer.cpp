/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/leaky_relu_layer.hpp"

#include <memory>
#include <stdexcept>

namespace tnn {

LeakyReLULayer::LeakyReLULayer(float negative_slope, const std::string &name)
    : StatelessLayer(name),
      activation_(std::make_unique<LeakyReLU>(negative_slope)),
      negative_slope_(negative_slope) {}

Tensor LeakyReLULayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  Tensor output = get_output_tensor(input->shape());

  if (this->is_training_) {
    // Cache boolean mask (1 byte per element) instead of full input
    Tensor mask = this->get_cache_tensor(input->shape(), DType_t::UINT8_T);
    set_mutable_cache(mb_id, "mask", mask);

    // Compute LeakyReLU and mask
    activation_->apply(input, output);

    // Compute mask: 1 where input > 0, 0 otherwise
    const size_t num_elements = input->size();
    if (input->device_type() == DeviceType::CPU) {
      const float *input_data = input->data_as<float>();
      uint8_t *mask_data = mask->data_as<uint8_t>();
      for (size_t i = 0; i < num_elements; ++i) {
        mask_data[i] = (input_data[i] > 0.0f) ? 1 : 0;
      }
    }
#ifdef USE_CUDA
    else if (input->device_type() == DeviceType::GPU) {
      throw std::runtime_error("LeakyReLULayer: GPU mask computation not yet implemented");
    }
#endif
  } else {
    activation_->apply(input, output);
  }

  return output;
}

Tensor LeakyReLULayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  const ConstTensor &mask = this->get_mutable_cache(mb_id, "mask");
  if (!mask) {
    throw std::runtime_error("No cached mask found for backward pass in LeakyReLULayer");
  }

  Tensor grad_input = get_output_tensor(grad_output->shape());

  // Gradient: grad_input = grad_output * (mask ? 1.0 : negative_slope)
  const size_t num_elements = grad_output->size();
  if (grad_output->device_type() == DeviceType::CPU) {
    const float *grad_out_data = grad_output->data_as<float>();
    const uint8_t *mask_data = mask->data_as<uint8_t>();
    float *grad_in_data = grad_input->data_as<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      float slope = mask_data[i] ? 1.0f : negative_slope_;
      grad_in_data[i] = grad_out_data[i] * slope;
    }
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    throw std::runtime_error("LeakyReLULayer: GPU backward not yet implemented");
  }
#endif

  return grad_input;
}

LayerConfig LeakyReLULayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("negative_slope", negative_slope_);
  return config;
}

std::unique_ptr<LeakyReLULayer> LeakyReLULayer::create_from_config(const LayerConfig &config) {
  float negative_slope = config.get<float>("negative_slope", 0.01f);
  return std::make_unique<LeakyReLULayer>(negative_slope, config.name);
}

}  // namespace tnn
