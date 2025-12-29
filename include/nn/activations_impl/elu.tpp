/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/elu.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/elu_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/elu_kernels.hpp"
#endif

namespace tnn {
template <typename T> ELU<T>::ELU(T alpha) : alpha_(alpha) {}

template <typename T>
std::unique_ptr<Task> ELU<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for ELU");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for ELU");
  }

  const T *input_data = input.data();
  T *output_data = output.data();
  const size_t size = input.size();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::elu<T>, input_data, output_data, size, alpha_);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::elu<T>, input_data, output_data, size, alpha_);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
std::unique_ptr<Task> ELU<T>::compute_gradient(const Tensor<T> &input, const Tensor<T> &grad_output,
                                               Tensor<T> &grad_input) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for ELU");
  }
  if (grad_output.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::elu_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size(), alpha_);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::elu_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size(), alpha_);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string ELU<T>::name() const { return "elu"; }

template <typename T> std::unique_ptr<EWActivationFunction<T>> ELU<T>::clone() const {
  return std::make_unique<ELU<T>>(*this);
}

// Explicit template instantiations
template class ELU<float>;
template class ELU<double>;

} // namespace tnn