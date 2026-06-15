/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/gelu.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace synet {

class GELULayerImpl : public SISOLayerImpl {
private:
  std::unique_ptr<GELU> activation_;

protected:
  Tensor forward_impl(const Tensor &input, Residuals &residualsuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residualsuals) override;

public:
  static constexpr const char *TYPE_NAME = "gelu";

  explicit GELULayerImpl(const std::string &name = "gelu");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<GELULayerImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }
};

class GELULayer : public LayerRef<GELULayerImpl> {
public:
  explicit GELULayer(const std::string &name = "gelu")
      : LayerRef(std::make_shared<GELULayerImpl>(name)) {}

  using LayerRef<GELULayerImpl>::LayerRef;
};

}  // namespace synet
