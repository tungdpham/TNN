/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
template <typename T = float> class GELU : public EWActivationFunction<T> {
public:
  std::unique_ptr<Task> apply(const Tensor<T> &input, Tensor<T> &output) const override;
  std::unique_ptr<Task> compute_gradient(const Tensor<T> &input, const Tensor<T> &grad_output,
                                         Tensor<T> &grad_input) const override;

  std::string name() const override { return "gelu"; }
  std::unique_ptr<EWActivationFunction<T>> clone() const override {
    return std::make_unique<GELU<T>>();
  }
};
} // namespace tnn

#include "nn/activations_impl/gelu.tpp"
