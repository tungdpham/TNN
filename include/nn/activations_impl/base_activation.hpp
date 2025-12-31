/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"
#include <memory>

namespace tnn {
template <typename T = float> class EWActivationFunction {
public:
  virtual ~EWActivationFunction() = default;

  virtual std::unique_ptr<Task> apply(const Tensor<T> &input, Tensor<T> &output) const = 0;

  virtual std::unique_ptr<Task> compute_gradient(const Tensor<T> &input,
                                                 const Tensor<T> &grad_output,
                                                 Tensor<T> &grad_input) const = 0;

  virtual std::string name() const = 0;
  virtual std::unique_ptr<EWActivationFunction<T>> clone() const = 0;
};
} // namespace tnn