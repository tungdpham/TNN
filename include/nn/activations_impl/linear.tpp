/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/linear.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {
template <typename T> void Linear<T>::apply(Tensor<T> &tensor) const { (void)tensor; }

template <typename T>
void Linear<T>::compute_gradient_inplace(const Tensor<T> &input,
                                         Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != input.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }
}

template <typename T> std::string Linear<T>::name() const { return "linear"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Linear<T>::clone() const {
  return std::make_unique<Linear<T>>(*this);
}

// Explicit template instantiations
template class Linear<float>;
template class Linear<double>;

} // namespace tnn