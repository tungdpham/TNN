/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/elu.hpp"

#include <cassert>

#include "nn/activations_impl/cpu/elu_kernels.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "nn/activations_impl/cuda/elu_kernels.hpp"
#endif

namespace tnn {
ELU::ELU(float alpha) : alpha_(alpha) {}

std::unique_ptr<Task> ELU::apply(const ConstTensor &input, Tensor &output) const {
  if (input->shape() != output->shape()) {
    throw std::runtime_error("Input and output shapes must match for ELU");
  }
  if (input->device() != output->device()) {
    throw std::runtime_error("Input and output must be on the same device for ELU");
  }

  DISPATCH_ON_DTYPE(input->data_type(), T, return apply_impl<T>(input, output, "default"));
}

std::unique_ptr<Task> ELU::compute_gradient(const ConstTensor &input,
                                            const ConstTensor &grad_output,
                                            Tensor &grad_input) const {
  assert(grad_output->shape() == grad_input->shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output->device() != grad_input->device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for ELU");
  }
  DISPATCH_ON_DTYPE(input->data_type(), T,
                    return compute_gradient_impl<T>(input, grad_output, grad_input, "default"));
}

std::string ELU::name() const { return "elu"; }

std::unique_ptr<ActivationFunction> ELU::clone() const { return std::make_unique<ELU>(alpha_); }

template <typename Compute_T>
std::unique_ptr<Task> ELU::apply_impl(const ConstTensor &input, Tensor &output,
                                      const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() || output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("ELU tensor dtype mismatch with dispatch type");
  }

  const size_t size = input->size();
  const Compute_T alpha_typed = static_cast<Compute_T>(alpha_);
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::elu<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), size, alpha_typed);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::elu<Compute_T>, input->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), size, alpha_typed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for ELU apply");
  }
  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task> ELU::compute_gradient_impl(const ConstTensor &input,
                                                 const ConstTensor &grad_output, Tensor &grad_input,
                                                 const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() ||
      grad_output->data_type() != dtype_of<Compute_T>() ||
      grad_input->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("ELU tensor dtype mismatch with dispatch type");
  }

  const size_t size = grad_output->size();
  const Compute_T alpha_typed = static_cast<Compute_T>(alpha_);
  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::elu_gradient<Compute_T>, input->data_as<Compute_T>(),
                           grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           size, alpha_typed);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::elu_gradient<Compute_T>, input->data_as<Compute_T>(),
                            grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                            size, alpha_typed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for ELU compute_gradient");
  }
  return nullptr;
}

}  // namespace tnn
