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

namespace tnn {

ReLULayer::ReLULayer(const std::string &name)
    : StatelessLayer(name),
      activation_(std::make_unique<ReLU>()) {}

Tensor ReLULayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  Tensor output = get_output_tensor(input->shape());
  const size_t num_elements = input->size();

  if (this->is_training_) {
    Tensor mask = this->get_cache_tensor(input->shape(), DType_t::UINT8_T);
    set_mutable_cache(mb_id, "mask", mask);

    // Fused kernel: compute ReLU and mask in a single pass
    if (input->device_type() == DeviceType::CPU) {
      DISPATCH_DTYPE(input->data_type(), T, {
        create_cpu_task(this->flow_handle_, cpu::relu::relu_forward_with_mask<T>,
                        input->data_as<T>(), output->data_as<T>(), mask->data_as<uint8_t>(),
                        num_elements);
      });
    }
#ifdef USE_CUDA
    else if (input->device_type() == DeviceType::GPU) {
      DISPATCH_DTYPE(input->data_type(), T, {
        create_cuda_task(this->flow_handle_, cuda::relu::relu_forward_with_mask<T>,
                         input->data_as<T>(), output->data_as<T>(), mask->data_as<uint8_t>(),
                         num_elements);
      });
    }
#endif
    else {
      throw std::runtime_error("ReLULayer: Unsupported device type");
    }
  } else {
    // Inference mode: just apply activation
    activation_->apply(input, output);
  }

  return output;
}

Tensor ReLULayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  const ConstTensor &mask = this->get_mutable_cache(mb_id, "mask");
  if (!mask) {
    throw std::runtime_error("No cached mask found for backward pass in ReLULayer");
  }

  Tensor grad_input = get_output_tensor(grad_output->shape());
  const size_t num_elements = grad_output->size();

  if (grad_output->device_type() == DeviceType::CPU) {
    DISPATCH_DTYPE(grad_output->data_type(), T, {
      create_cpu_task(this->flow_handle_, cpu::relu::relu_backward_with_mask<T>,
                      grad_output->data_as<T>(), grad_input->data_as<T>(), mask->data_as<uint8_t>(),
                      num_elements);
    });
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    DISPATCH_DTYPE(grad_output->data_type(), T, {
      create_cuda_task(this->flow_handle_, cuda::relu::relu_backward_with_mask<T>,
                       grad_output->data_as<T>(), grad_input->data_as<T>(),
                       mask->data_as<uint8_t>(), num_elements);
    });
  }
#endif
  else {
    throw std::runtime_error("ReLULayer: Unsupported device type");
  }

  return grad_input;
}

LayerConfig ReLULayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::unique_ptr<ReLULayer> ReLULayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<ReLULayer>(config.name);
}

}  // namespace tnn
