/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/leaky_relu.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "nn/activations_impl/cpu/leaky_relu_kernels.hpp"
#ifdef USE_CUDA
#include "nn/activations_impl/cuda/leaky_relu_kernels.hpp"
#endif

namespace tnn {
LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

std::unique_ptr<Task> LeakyReLU::apply(const Tensor &input, Tensor &output) const {
  if (input->shape() != output->shape()) {
    throw std::runtime_error("Input and output shapes must match for LeakyReLU");
  }
  if (input->device() != output->device()) {
    throw std::runtime_error("Input and output must be on the same device for LeakyReLU");
  }

  switch (input->data_type()) {
  case DType_t::FP32:
    return apply_impl<float>(input, output, "default");
  case DType_t::FP64:
    return apply_impl<double>(input, output, "default");
  case DType_t::FP16:
    return apply_impl<fp16>(input, output, "default");
  default:
    throw std::runtime_error("Unsupported data type for LeakyReLU apply");
  }
}

std::unique_ptr<Task> LeakyReLU::compute_gradient(const Tensor &input, const Tensor &grad_output,
                                                  Tensor &grad_input) const {
  assert(grad_output->shape() == grad_input->shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output->device() != grad_input->device()) {
    throw std::runtime_error(
        "Input and upstream gradient must be on the same device for LeakyReLU");
  }
  switch (grad_output->data_type()) {
  case DType_t::FP32:
    return compute_gradient_impl<float>(input, grad_output, grad_input, "default");
  case DType_t::FP64:
    return compute_gradient_impl<double>(input, grad_output, grad_input, "default");
  case DType_t::FP16:
    return compute_gradient_impl<fp16>(input, grad_output, grad_input, "default");
  default:
    throw std::runtime_error("Unsupported data type for LeakyReLU compute_gradient");
  }
}

std::string LeakyReLU::name() const { return "leaky_relu"; }

std::unique_ptr<ActivationFunction> LeakyReLU::clone() const {
  return std::make_unique<LeakyReLU>(negative_slope_);
}

template <typename Compute_T>
std::unique_ptr<Task> LeakyReLU::apply_impl(const Tensor &input, Tensor &output,
                                            const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() || output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("LeakyReLU tensor dtype mismatch with dispatch type");
  }

  const size_t size = input->size();
  const Compute_T slope_typed = static_cast<Compute_T>(negative_slope_);
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::leaky_relu<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), size, slope_typed);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::leaky_relu<Compute_T>, input->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), size, slope_typed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for LeakyReLU apply");
  }
  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task>
LeakyReLU::compute_gradient_impl(const Tensor &input, const Tensor &grad_output, Tensor &grad_input,
                                 const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() ||
      grad_output->data_type() != dtype_of<Compute_T>() ||
      grad_input->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("LeakyReLU tensor dtype mismatch with dispatch type");
  }

  const size_t size = grad_output->size();
  const Compute_T slope_typed = static_cast<Compute_T>(negative_slope_);
  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::leaky_relu_gradient<Compute_T>,
                           input->data_as<Compute_T>(), grad_output->data_as<Compute_T>(),
                           grad_input->data_as<Compute_T>(), size, slope_typed);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::leaky_relu_gradient<Compute_T>,
                            input->data_as<Compute_T>(), grad_output->data_as<Compute_T>(),
                            grad_input->data_as<Compute_T>(), size, slope_typed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for LeakyReLU compute_gradient");
  }
  return nullptr;
}

} // namespace tnn
