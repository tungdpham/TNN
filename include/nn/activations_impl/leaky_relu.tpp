/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/leaky_relu.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/leaky_relu_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/leaky_relu_kernels.hpp"
#endif

namespace tnn {
template <typename T> LeakyReLU<T>::LeakyReLU(T negative_slope) : negative_slope_(negative_slope) {}

template <typename T>
std::unique_ptr<Task> LeakyReLU<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for LeakyReLU");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for LeakyReLU");
  }

  const T *input_data = input.data();
  T *output_data = output.data();
  const size_t size = input.size();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::leaky_relu<T>, input_data, output_data, size,
                           negative_slope_);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::leaky_relu<T>, input_data, output_data, size,
                           negative_slope_);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
std::unique_ptr<Task> LeakyReLU<T>::compute_gradient(const Tensor<T> &input,
                                                     const Tensor<T> &grad_output,
                                                     Tensor<T> &grad_input) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error(
        "Input and upstream gradient must be on the same device for LeakyReLU");
  }
  if (grad_output.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::leaky_relu_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), grad_output.size(), negative_slope_);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::leaky_relu_gradient<T>, input.data(),
                           grad_output.data(), grad_input.data(), grad_output.size(),
                           negative_slope_);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string LeakyReLU<T>::name() const { return "leaky_relu"; }

template <typename T> std::unique_ptr<EWActivationFunction<T>> LeakyReLU<T>::clone() const {
  return std::make_unique<LeakyReLU<T>>(*this);
}

template class LeakyReLU<float>;
template class LeakyReLU<double>;

} // namespace tnn