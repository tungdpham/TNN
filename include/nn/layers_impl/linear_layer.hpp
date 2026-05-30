/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/linear.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class LinearLayerImpl : public SISOLayerImpl {
private:
  std::unique_ptr<Linear> activation_;

protected:
  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "linear";

  explicit LinearLayerImpl(const std::string &name = "linear");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<LinearLayerImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }
};

class LinearLayer : public LayerRef<LinearLayerImpl> {
public:
  explicit LinearLayer(const std::string &name = "linear")
      : LayerRef(std::make_shared<LinearLayerImpl>(name)) {}

  using LayerRef<LinearLayerImpl>::LayerRef;
};

}  // namespace tnn
