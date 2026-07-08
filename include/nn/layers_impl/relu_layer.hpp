/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class ReLULayerImpl : public SISOLayerImpl {
protected:
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "relu";

  explicit ReLULayerImpl(const std::string &name = "relu");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<ReLULayerImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }
};

class ReLULayer : public LayerRef<ReLULayerImpl> {
public:
  explicit ReLULayer(const std::string &name = "relu")
      : LayerRef(std::make_shared<ReLULayerImpl>(name)) {}

  using LayerRef<ReLULayerImpl>::LayerRef;
};

}  // namespace tunx