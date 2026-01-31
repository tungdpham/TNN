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
class ReLU : public ActivationFunction {
public:
  explicit ReLU();

  std::unique_ptr<Task> apply(const Tensor &input, Tensor &output) const override;

  std::unique_ptr<Task> compute_gradient(const Tensor &input, const Tensor &grad_output,
                                         Tensor &grad_input) const override;

  std::string name() const override;
  std::unique_ptr<ActivationFunction> clone() const override;

private:
  template <typename Compute_T>
  std::unique_ptr<Task> apply_impl(const Tensor &input, Tensor &output,
                                   const std::string &flow_id) const;

  template <typename Compute_T>
  std::unique_ptr<Task> compute_gradient_impl(const Tensor &input, const Tensor &grad_output,
                                              Tensor &grad_input, const std::string &flow_id) const;
};

}  // namespace tnn
