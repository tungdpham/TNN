/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/gelu.hpp"

#include <cassert>

#include "nn/activations_impl/cpu/gelu_kernels.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "nn/activations_impl/cuda/gelu_kernels.hpp"
#endif

namespace tnn {

std::unique_ptr<Task> GELU::apply(const ConstTensor &input, const Tensor &output) const {
  if (input->shape() != output->shape()) {
    throw std::runtime_error("Input and output shapes must match for GELU");
  }
  if (input->device() != output->device()) {
    throw std::runtime_error("Input and output must be on the same device for GELU");
  }

  DISPATCH_ON_DTYPE(input->data_type(), T, return apply_impl<T>(input, output, "default"));
}

std::unique_ptr<Task> GELU::compute_gradient(const ConstTensor &input,
                                             const ConstTensor &grad_output,
                                             const Tensor &grad_input) const {
  assert(grad_output->shape() == grad_input->shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output->device() != grad_input->device()) {
    throw std::runtime_error("Tensors must be on the same device for GELU");
  }

  DISPATCH_ON_DTYPE(input->data_type(), T,
                    return compute_gradient_impl<T>(input, grad_output, grad_input, "default"));
}

template <typename Compute_T>
std::unique_ptr<Task> GELU::apply_impl(const ConstTensor &input, const Tensor &output,
                                       const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() || output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("GELU tensor dtype mismatch with dispatch type");
  }

  const size_t size = input->size();
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::gelu<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), size);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::gelu<Compute_T>, input->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for GELU apply");
  }
  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task> GELU::compute_gradient_impl(const ConstTensor &input,
                                                  const ConstTensor &grad_output,
                                                  const Tensor &grad_input,
                                                  const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() ||
      grad_output->data_type() != dtype_of<Compute_T>() ||
      grad_input->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("GELU tensor dtype mismatch with dispatch type");
  }

  const size_t size = input->size();
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::gelu_gradient<Compute_T>, input->data_as<Compute_T>(),
                           grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           size);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::gelu_gradient<Compute_T>, input->data_as<Compute_T>(),
                            grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for GELU compute_gradient");
  }
  return nullptr;
}

}  // namespace tnn
