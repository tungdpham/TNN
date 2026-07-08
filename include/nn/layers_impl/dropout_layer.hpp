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

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class DropoutLayerImpl : public SISOLayerImpl {
private:
  float dropout_rate_;
  mutable std::mt19937 generator_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit DropoutLayerImpl(float dropout_rate, const std::string &name = "dropout");

  static constexpr const char *TYPE_NAME = "dropout";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<DropoutLayerImpl> create_from_config(const LayerConfig &config);
};

class DropoutLayer : public LayerRef<DropoutLayerImpl> {
public:
  explicit DropoutLayer(float dropout_rate, const std::string &name = "dropout")
      : LayerRef(std::make_shared<DropoutLayerImpl>(dropout_rate, name)) {}

  using LayerRef<DropoutLayerImpl>::LayerRef;
};

}  // namespace tunx
