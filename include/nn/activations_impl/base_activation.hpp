/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>

#include "tensor/tensor.hpp"

namespace tnn {
class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;

  virtual std::unique_ptr<Task> apply(const ConstTensor &input, const Tensor &output) const = 0;

  virtual std::unique_ptr<Task> compute_gradient(const ConstTensor &input,
                                                 const ConstTensor &grad_output,
                                                 const Tensor &grad_input) const = 0;

  virtual std::string name() const = 0;
  virtual std::unique_ptr<ActivationFunction> clone() const = 0;
};
}  // namespace tnn