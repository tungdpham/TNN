/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/linear.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {
template <typename T>
std::unique_ptr<Task> Linear<T>::apply(const Tensor<T> &input, Tensor<T> &output) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for Linear");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for Linear");
  }
  // Linear activation is just identity, copy input to output
  if (&input != &output) {
    return ops::copy(input.data_ptr(), output.data_ptr(), input.size());
  }
  return nullptr;
}

template <typename T>
std::unique_ptr<Task> Linear<T>::compute_gradient(const Tensor<T> &input,
                                                  const Tensor<T> &grad_output,
                                                  Tensor<T> &grad_input) const {
  if (grad_input.shape() != grad_output.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }
  return ops::copy(grad_output.data_ptr(), grad_input.data_ptr(), grad_output.size());
}

template <typename T> std::string Linear<T>::name() const { return "linear"; }

template <typename T> std::unique_ptr<EWActivationFunction<T>> Linear<T>::clone() const {
  return std::make_unique<Linear<T>>(*this);
}

// Explicit template instantiations
template class Linear<float>;
template class Linear<double>;

} // namespace tnn