/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/relu_layer.hpp"

#include <memory>
#include <stdexcept>

#include "device/task.hpp"
#include "nn/layers_impl/cpu/relu_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/relu_ops.hpp"
#endif

namespace synet {

ReLULayerImpl::ReLULayerImpl(const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<ReLU>()) {}

Tensor ReLULayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor output = get_tensor(input.shape(), io_dtype_);
  size_t num_elements = input.size();

  if (this->is_training_) {
    Tensor mask = this->get_tensor(input.shape(), DType_t::UINT8_T);
    residuals["mask"] = mask;

    // Fused kernel: compute ReLU and mask in a single pass
    if (input.device_type() == DeviceType::CPU) {
      DISPATCH_DTYPE(input.data_type(), T, {
        create_cpu_task(this->flow_handle_, cpu::relu::relu_forward_with_mask<T>,
                        input.data_as<T>(), output.data_as<T>(), mask.data_as<uint8_t>(),
                        num_elements);
      });
    }
#ifdef USE_CUDA
    else if (input.device_type() == DeviceType::GPU) {
      DISPATCH_DTYPE(input.data_type(), T, {
        create_cuda_task(this->flow_handle_, cuda::relu::relu_forward_with_mask<T>,
                         input.data_as<T>(), output.data_as<T>(), mask.data_as<uint8_t>(),
                         num_elements);
      });
    }
#endif
    else {
      throw std::runtime_error("ReLULayerImpl: Unsupported device type");
    }
  } else {
    // Inference mode: just apply activation
    activation_->apply(input, output);
  }

  return output;
}

Tensor ReLULayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor &mask = residuals["mask"];
  if (!mask) {
    throw std::runtime_error("No cached mask found for backward pass in ReLULayerImpl");
  }

  Tensor grad_input = get_tensor(grad_output.shape(), io_dtype_);
  size_t num_elements = grad_output.size();

  if (grad_output.device_type() == DeviceType::CPU) {
    DISPATCH_DTYPE(grad_output.data_type(), T, {
      create_cpu_task(this->flow_handle_, cpu::relu::relu_backward_with_mask<T>,
                      grad_output.data_as<T>(), grad_input.data_as<T>(), mask.data_as<uint8_t>(),
                      num_elements);
    });
  }
#ifdef USE_CUDA
  else if (grad_output.device_type() == DeviceType::GPU) {
    DISPATCH_DTYPE(grad_output.data_type(), T, {
      create_cuda_task(this->flow_handle_, cuda::relu::relu_backward_with_mask<T>,
                       grad_output.data_as<T>(), grad_input.data_as<T>(), mask.data_as<uint8_t>(),
                       num_elements);
    });
  }
#endif
  else {
    throw std::runtime_error("ReLULayerImpl: Unsupported device type");
  }

  return grad_input;
}

LayerConfig ReLULayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<ReLULayerImpl> ReLULayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<ReLULayerImpl>(config.name);
}

}  // namespace synet
