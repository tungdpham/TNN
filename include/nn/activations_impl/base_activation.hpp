/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"

namespace tnn {
template <typename T = float> class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;

  virtual void apply(Tensor<T> &tensor) const = 0;

  Tensor<T> compute_gradient(const Tensor<T> &input,
                             const Tensor<T> *upstream_gradient = nullptr) const {
    Tensor<T> gradient;
    assert(upstream_gradient != nullptr && "Upstream gradient must be provided");
    gradient = upstream_gradient->clone();
    compute_gradient_inplace(input, gradient);
    return gradient;
  }

  virtual void compute_gradient_inplace(const Tensor<T> &input,
                                        Tensor<T> &upstream_gradient) const = 0;

  virtual std::string name() const = 0;
  virtual std::unique_ptr<ActivationFunction<T>> clone() const = 0;
};
} // namespace tnn