/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <random>
#include <string>

#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class DropoutLayer : public StatelessLayer {
private:
  float dropout_rate_;
  mutable std::mt19937 generator_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const ConstTensor &input, const Tensor &output,
                                    const Tensor &mask, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                     const ConstTensor &mask, flowHandle_t handle) const;

  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  explicit DropoutLayer(float dropout_rate, const std::string &name = "dropout");

  static constexpr const char *TYPE_NAME = "dropout";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::unique_ptr<DropoutLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
