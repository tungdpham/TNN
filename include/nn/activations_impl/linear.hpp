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
class Linear : public ActivationFunction {
public:
  std::unique_ptr<Task> apply(const ConstTensor &input, const Tensor &output) const override;

  std::unique_ptr<Task> compute_gradient(const ConstTensor &input, const ConstTensor &grad_output,
                                         const Tensor &grad_input) const override;

  std::string name() const override;
  std::unique_ptr<ActivationFunction> clone() const override;
};

}  // namespace tnn
