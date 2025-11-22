/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/softmax.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/softmax_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/softmax_kernels.hpp"
#endif

namespace tnn {
template <typename T> void Softmax<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t batch_size = tensor.batch_size();
  const size_t channels = tensor.channels();
  const size_t height = tensor.height();
  const size_t width = tensor.width();

  if (tensor.device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::softmax<T>, data, data, batch_size, channels, height, width);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::softmax<T>, data, data, batch_size, channels, height, width);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
void Softmax<T>::compute_gradient_inplace(const Tensor<T> &input,
                                          Tensor<T> &upstream_gradient) const {
  assert(input.shape() == upstream_gradient.shape() &&
         "Shapes must match for in-place gradient computation");
  if (input.device() != upstream_gradient.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for Softmax");
  }

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();

  if (input.device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::softmax_gradient<T>, input.data(), upstream_gradient.data(),
                    batch_size, channels, height, width);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::softmax_gradient<T>, input.data(), upstream_gradient.data(),
                    batch_size, channels, height, width);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string Softmax<T>::name() const { return "softmax"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Softmax<T>::clone() const {
  return std::make_unique<Softmax<T>>(*this);
}

// Explicit template instantiations
template class Softmax<float>;
template class Softmax<double>;

} // namespace tnn