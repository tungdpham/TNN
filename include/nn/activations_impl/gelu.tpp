/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/gelu.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/gelu_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/gelu_kernels.hpp"
#endif

namespace tnn {

template <typename T>
std::unique_ptr<Task> GELU<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for GELU");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for GELU");
  }

  const T *input_data = input.data();
  T *output_data = output.data();
  const size_t size = input.size();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::gelu<T>, input_data, output_data, size);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::gelu<T>, input_data, output_data, size);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
std::unique_ptr<Task> GELU<T>::compute_gradient(const Tensor<T> &input,
                                                const Tensor<T> &grad_output,
                                                Tensor<T> &grad_input) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Tensors must be on the same device");
  }

  const T *input_data = input.data();
  const T *grad_out_data = grad_output.data();
  T *grad_in_data = grad_input.data();
  const size_t size = input.size();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::gelu_gradient<T>, input_data, grad_out_data,
                           grad_in_data, size);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::gelu_gradient<T>, input_data, grad_out_data,
                           grad_in_data, size);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

} // namespace tnn
