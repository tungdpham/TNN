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
template <typename T = float> class ReLU : public ActivationFunction<T> {
public:
  explicit ReLU();

  std::unique_ptr<Task> apply(Tensor<T> &tensor) const override;

  std::unique_ptr<Task> compute_gradient_inplace(const Tensor<T> &input,
                                                 Tensor<T> &upstream_gradient) const override;

  std::string name() const override;
  std::unique_ptr<ActivationFunction<T>> clone() const override;
};

} // namespace tnn

#include "nn/activations_impl/relu.tpp"