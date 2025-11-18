/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/elu.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {
template <typename T> ELU<T>::ELU(T alpha) : alpha_(alpha) {}

template <typename T> void ELU<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  parallel_for<size_t>(0, size, [&](size_t i) {
    data[i] = data[i] > T(0) ? data[i] : alpha_ * (std::exp(data[i]) - T(1));
  });
}

template <typename T>
void ELU<T>::compute_gradient_inplace(const Tensor<T> &input, Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != input.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  const T *input_data = input.data();
  T *grad_data = upstream_gradient.data();
  size_t size = input.size();

  parallel_for<size_t>(0, size, [&](size_t i) {
    T local_grad = input_data[i] > T(0) ? T(1) : alpha_ * std::exp(input_data[i]);
    grad_data[i] *= local_grad;
  });
}

template <typename T> std::string ELU<T>::name() const { return "elu"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> ELU<T>::clone() const {
  return std::make_unique<ELU<T>>(*this);
}

// Explicit template instantiations
template class ELU<float>;
template class ELU<double>;

} // namespace tnn