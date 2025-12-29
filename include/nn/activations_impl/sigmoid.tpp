/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/sigmoid.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/sigmoid_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/sigmoid_kernels.hpp"
#endif

namespace tnn {
template <typename T>
std::unique_ptr<Task> Sigmoid<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for Sigmoid");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for Sigmoid");
  }

  const T *input_data = input.data();
  T *output_data = output.data();
  const size_t size = input.size();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::sigmoid<T>, input_data, output_data, size);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::sigmoid<T>, input_data, output_data, size);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
std::unique_ptr<Task> Sigmoid<T>::compute_gradient(const Tensor<T> &input,
                                                   const Tensor<T> &grad_output,
                                                   Tensor<T> &grad_input) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for Sigmoid");
  }
  if (grad_output.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::sigmoid_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size());
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::sigmoid_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size());
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string Sigmoid<T>::name() const { return "sigmoid"; }

template <typename T> std::unique_ptr<EWActivationFunction<T>> Sigmoid<T>::clone() const {
  return std::make_unique<Sigmoid<T>>(*this);
}

// Explicit template instantiations
template class Sigmoid<float>;
template class Sigmoid<double>;

} // namespace tnn