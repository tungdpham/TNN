/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/tanh.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cassert>

#include "cpu/tanh_kernels.hpp"
#ifdef USE_CUDA
#include "cuda/tanh_kernels.hpp"
#endif

namespace tnn {

template <typename T> void Tanh<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  if (tensor.device_type() == DeviceType::CPU) {
    ops::create_cpu_task(tensor.device(), cpu::tanh<T>, data, data, size);
  } else {
#ifdef USE_CUDA
    ops::create_gpu_task(tensor.device(), cuda::tanh<T>, data, data, size);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
void Tanh<T>::compute_gradient_inplace(const Tensor<T> &input, Tensor<T> &upstream_gradient) const {
  assert(input.shape() == upstream_gradient.shape() &&
         "Shapes must match for in-place gradient computation");
  if (input.device() != upstream_gradient.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for Tanh");
  }
  if (input.device_type() == DeviceType::CPU) {
    ops::create_cpu_task(input.device(), cpu::tanh_gradient<T>, input.data(),
                         upstream_gradient.data(), input.size());
  } else {
#ifdef USE_CUDA
    ops::create_gpu_task(input.device(), cuda::tanh_gradient<T>, input.data(),
                         upstream_gradient.data(), input.size());
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string Tanh<T>::name() const { return "tanh"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Tanh<T>::clone() const {
  return std::make_unique<Tanh<T>>(*this);
}

// Explicit template instantiations
template class Tanh<float>;
template class Tanh<double>;

} // namespace tnn