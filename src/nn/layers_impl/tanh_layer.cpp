/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/tanh_layer.hpp"

#include <memory>
#include <stdexcept>

namespace tunx {

TanhLayerImpl::TanhLayerImpl(const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<Tanh>()) {}

Tensor TanhLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor output = get_tensor(input.shape(), io_dtype_);
  activation_->apply(input, output);

  if (this->is_training_) {
    // tanh'(x) = 1 - tanh(x)^2
    residuals["output"] = output;
  }

  return output;
}

Tensor TanhLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor &output = residuals["output"];
  if (!output) {
    throw std::runtime_error("No cached output found for backward pass in TanhLayerImpl");
  }

  Tensor grad_input = get_tensor(grad_output.shape(), io_dtype_);

  // Gradient: grad_input = grad_output * (1 - output^2)
  size_t num_elements = grad_output.size();
  if (grad_output.device_type() == DeviceType::CPU) {
    const float *grad_out_data = grad_output.data_as<float>();
    const float *output_data = output.data_as<float>();
    float *grad_in_data = grad_input.data_as<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      float tanh_val = output_data[i];
      grad_in_data[i] = grad_out_data[i] * (1.0f - tanh_val * tanh_val);
    }
  }
#ifdef USE_CUDA
  else if (grad_output.device_type() == DeviceType::CUDA) {
    throw std::runtime_error("TanhLayerImpl: CUDA backward not yet implemented");
  }
#endif

  return grad_input;
}

LayerConfig TanhLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<TanhLayerImpl> TanhLayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<TanhLayerImpl>(config.name);
}

}  // namespace tunx
