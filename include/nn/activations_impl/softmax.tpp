/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/softmax.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/softmax_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/softmax_kernels.hpp"
#endif

namespace tnn {
template <typename T>
std::unique_ptr<Task> Softmax<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for Softmax");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for Softmax");
  }

  const T *input_data = input.data();
  T *output_data = output.data();
  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::softmax<T>, input_data, output_data, batch_size,
                           channels, height, width);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::softmax<T>, input_data, output_data, batch_size,
                           channels, height, width);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
std::unique_ptr<Task> Softmax<T>::compute_gradient(const Tensor<T> &input,
                                                   const Tensor<T> &grad_output,
                                                   Tensor<T> &grad_input) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place gradient computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for Softmax");
  }

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();

  if (grad_output.device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::softmax_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), batch_size, channels, height, width);
  } else {
#ifdef USE_CUDA
    return create_gpu_task("default", cuda::softmax_gradient<T>, input.data(), grad_output.data(),
                           grad_input.data(), batch_size, channels, height, width);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string Softmax<T>::name() const { return "softmax"; }

template <typename T> std::unique_ptr<Softmax<T>> Softmax<T>::clone() const {
  return std::make_unique<Softmax<T>>(*this);
}

// Explicit template instantiations
template class Softmax<float>;
template class Softmax<double>;

} // namespace tnn