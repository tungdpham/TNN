/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/relu.hpp"

#include <cassert>

#include "nn/activations_impl/cpu/relu_kernels.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "nn/activations_impl/cuda/relu_kernels.hpp"
#endif

namespace tnn {
ReLU::ReLU() {}

std::unique_ptr<Task> ReLU::apply(const ConstTensor &input, const Tensor &output) const {
  if (input->shape() != output->shape()) {
    throw std::runtime_error("Input and output shapes must match for ReLU");
  }
  if (input->device() != output->device()) {
    throw std::runtime_error("Input and output must be on the same device for ReLU");
  }

  DISPATCH_DTYPE(input->data_type(), T, return apply_impl<T>(input, output, defaultFlowHandle));
}

std::unique_ptr<Task> ReLU::compute_gradient(const ConstTensor &input,
                                             const ConstTensor &grad_output,
                                             const Tensor &grad_input) const {
  assert(grad_output->shape() == grad_input->shape() &&
         "Shapes must match for in-place grad_output computation");
  if (grad_output->device() != grad_input->device()) {
    throw std::runtime_error("Input and upstream grad_output must be on the same device for RELU");
  }
  DISPATCH_DTYPE(
      input->data_type(), T,
      return compute_gradient_impl<T>(input, grad_output, grad_input, defaultFlowHandle));
}

std::string ReLU::name() const { return "relu"; }

std::unique_ptr<ActivationFunction> ReLU::clone() const { return std::make_unique<ReLU>(*this); }

template <typename Compute_T>
std::unique_ptr<Task> ReLU::apply_impl(const ConstTensor &input, const Tensor &output,
                                       flowHandle_t handle) const {
  if (input->data_type() != dtype_of<Compute_T>() || output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("ReLU tensor dtype mismatch with dispatch type");
  }

  const size_t size = input->size();
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::relu<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), size);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::relu<Compute_T>, input->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for ReLU apply");
  }
  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task> ReLU::compute_gradient_impl(const ConstTensor &input,
                                                  const ConstTensor &grad_output,
                                                  const Tensor &grad_input,
                                                  flowHandle_t handle) const {
  if (input->data_type() != dtype_of<Compute_T>() ||
      grad_output->data_type() != dtype_of<Compute_T>() ||
      grad_input->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("ReLU tensor dtype mismatch with dispatch type");
  }

  const size_t size = grad_output->size();
  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::relu_gradient<Compute_T>, input->data_as<Compute_T>(),
                           grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           size);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::relu_gradient<Compute_T>, input->data_as<Compute_T>(),
                            grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for ReLU compute_gradient");
  }
  return nullptr;
}

}  // namespace tnn