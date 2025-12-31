/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "tensor/tensor.hpp"

namespace tnn {
template <typename T = float> class Softmax {
public:
  std::unique_ptr<Task> apply(const Tensor<T> &input, Tensor<T> &output) const;

  std::unique_ptr<Task> compute_gradient(const Tensor<T> &input, const Tensor<T> &grad_output,
                                         Tensor<T> &grad_input) const;

  std::string name() const;
  std::unique_ptr<Softmax<T>> clone() const;
};

} // namespace tnn

#include "nn/activations_impl/softmax.tpp"