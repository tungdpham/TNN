/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/sigmoid.hpp"
#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

/**
 * Sigmoid Layer with output caching
 * Caches the output activation during forward pass for efficient gradient computation.
 * Sigmoid gradient: grad_input = grad_output * output * (1 - output)
 */
class SigmoidLayer : public StatelessLayer {
private:
  std::unique_ptr<Sigmoid> activation_;

protected:
  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "sigmoid";

  explicit SigmoidLayer(const std::string &name = "sigmoid");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<SigmoidLayer> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }
};

}  // namespace tnn
