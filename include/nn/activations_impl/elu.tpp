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

template <typename T> void ELU<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  if (tensor.device_type() == DeviceType::CPU) {
    ops::create_cpu_task(tensor.device(), cpu::elu<T>, data, data, size, alpha_);
  } else {
#ifdef USE_CUDA
    ops::create_gpu_task(tensor.device(), cuda::elu<T>, data, data, size, alpha_);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T>
void ELU<T>::compute_gradient_inplace(const Tensor<T> &input, Tensor<T> &upstream_gradient) const {
  assert(input.shape() == upstream_gradient.shape() &&
         "Shapes must match for in-place gradient computation");
  if (input.device() != upstream_gradient.device()) {
    throw std::runtime_error("Input and upstream gradient must be on the same device for ELU");
  }
  if (input.device_type() == DeviceType::CPU) {
    ops::create_cpu_task(input.device(), cpu::elu_gradient<T>, input.data(),
                         upstream_gradient.data(), input.size(), alpha_);
  } else {
#ifdef USE_CUDA
    ops::create_gpu_task(input.device(), cuda::elu_gradient<T>, input.data(),
                         upstream_gradient.data(), input.size(), alpha_);
#else
    throw std::runtime_error("CUDA support is not enabled.");
#endif
  }
}

template <typename T> std::string ELU<T>::name() const { return "elu"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> ELU<T>::clone() const {
  return std::make_unique<ELU<T>>(*this);
}

// Explicit template instantiations
template class ELU<float>;
template class ELU<double>;

} // namespace tnn