/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/tanh.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "nn/activations_impl/cpu/tanh_kernels.hpp"
#ifdef USE_CUDA
#include "nn/activations_impl/cuda/tanh_kernels.hpp"
#endif

namespace tnn {

std::unique_ptr<Task> Tanh::apply(const Tensor &input, Tensor &output) const {
  if (input->shape() != output->shape()) {
    throw std::runtime_error("Input and output shapes must match for Tanh");
  }
  if (input->device() != output->device()) {
    throw std::runtime_error("Input and output must be on the same device for Tanh");
  }

  switch (input->data_type()) {
  case DType_t::FP32:
    return apply_impl<float>(input, output, "default");
  case DType_t::FP64:
    return apply_impl<double>(input, output, "default");
  case DType_t::FP16:
    return apply_impl<fp16>(input, output, "default");
  default:
    throw std::runtime_error("Unsupported data type for Tanh apply");
  }
}

std::unique_ptr<Task> Tanh::compute_gradient(const Tensor &input, const Tensor &grad_output,
                                             Tensor &grad_input) const {
  assert(grad_output->shape() == grad_input->shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output->device() != grad_input->device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for Tanh");
  }
  switch (grad_output->data_type()) {
  case DType_t::FP32:
    return compute_gradient_impl<float>(input, grad_output, grad_input, "default");
  case DType_t::FP64:
    return compute_gradient_impl<double>(input, grad_output, grad_input, "default");
  case DType_t::FP16:
    return compute_gradient_impl<fp16>(input, grad_output, grad_input, "default");
  default:
    throw std::runtime_error("Unsupported data type for Tanh compute_gradient");
  }
}

std::string Tanh::name() const { return "tanh"; }

std::unique_ptr<ActivationFunction> Tanh::clone() const { return std::make_unique<Tanh>(); }

template <typename Compute_T>
std::unique_ptr<Task> Tanh::apply_impl(const Tensor &input, Tensor &output,
                                       const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() || output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("Tanh tensor dtype mismatch with dispatch type");
  }

  const size_t size = input->size();
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::tanh<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), size);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::tanh<Compute_T>, input->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for Tanh apply");
  }
  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task> Tanh::compute_gradient_impl(const Tensor &input, const Tensor &grad_output,
                                                  Tensor &grad_input,
                                                  const std::string &flow_id) const {
  if (input->data_type() != dtype_of<Compute_T>() ||
      grad_output->data_type() != dtype_of<Compute_T>() ||
      grad_input->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("Tanh tensor dtype mismatch with dispatch type");
  }

  const size_t size = grad_output->size();
  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::tanh_gradient<Compute_T>, input->data_as<Compute_T>(),
                           grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           size);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::tanh_gradient<Compute_T>, input->data_as<Compute_T>(),
                            grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for Tanh compute_gradient");
  }
  return nullptr;
}

} // namespace tnn