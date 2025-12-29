/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/relu.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/relu_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/relu_kernels.hpp"
#endif

namespace tnn {
template <typename T> ReLU<T>::ReLU() {}

template <typename T>
std::unique_ptr<Task> ReLU<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for ReLU");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for ReLU");
  }

  const T *input_data = input.data();
  T *output_data = output.data();
  const size_t size = input.size();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::relu<T>, input_data, output_data, size);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::relu<T>, input_data, output_data, size);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
std::unique_ptr<Task> ReLU<T>::compute_gradient(const Tensor<T> &input,
                                                const Tensor<T> &grad_output,
                                                Tensor<T> &grad_input) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for RELU");
  }
  if (grad_output.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::relu_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size());
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::relu_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size());
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string ReLU<T>::name() const { return "relu"; }

template <typename T> std::unique_ptr<EWActivationFunction<T>> ReLU<T>::clone() const {
  return std::make_unique<ReLU<T>>(*this);
}

// Explicit template instantiations
template class ReLU<float>;
template class ReLU<double>;

} // namespace tnn