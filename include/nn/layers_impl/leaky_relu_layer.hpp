/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/leaky_relu.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class LeakyReLULayerImpl : public SISOLayerImpl {
private:
  std::unique_ptr<LeakyReLU> activation_;
  float negative_slope_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "leaky_relu";

  explicit LeakyReLULayerImpl(float negative_slope = 0.01f, const std::string &name = "leaky_relu");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<LeakyReLULayerImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }

  float get_negative_slope() const { return negative_slope_; }
};

class LeakyReLULayer : public LayerRef<LeakyReLULayerImpl> {
public:
  explicit LeakyReLULayer(float negative_slope = 0.01f, const std::string &name = "leaky_relu")
      : LayerRef(std::make_shared<LeakyReLULayerImpl>(negative_slope, name)) {}

  using LayerRef<LeakyReLULayerImpl>::LayerRef;
};

}  // namespace tunx
