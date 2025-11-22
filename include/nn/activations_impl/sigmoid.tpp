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
template <typename T> void Sigmoid<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  if (tensor.device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::sigmoid<T>, data, data, size);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::sigmoid<T>, data, data, size);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
void Sigmoid<T>::compute_gradient_inplace(const Tensor<T> &input,
                                          Tensor<T> &upstream_gradient) const {
  assert(input.shape() == upstream_gradient.shape() &&
         "Shapes must match for in-place gradient computation");
  if (input.device() != upstream_gradient.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for Sigmoid");
  }
  if (input.device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::sigmoid_gradient<T>, input.data(), upstream_gradient.data(),
                    input.size());
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::sigmoid_gradient<T>, input.data(), upstream_gradient.data(),
                    input.size());
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string Sigmoid<T>::name() const { return "sigmoid"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Sigmoid<T>::clone() const {
  return std::make_unique<Sigmoid<T>>(*this);
}

// Explicit template instantiations
template class Sigmoid<float>;
template class Sigmoid<double>;

} // namespace tnn