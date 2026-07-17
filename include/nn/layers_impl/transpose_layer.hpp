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

class TransposeLayerImpl : public SISOLayerImpl {
private:
  size_t dim0_, dim1_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  TransposeLayerImpl(size_t dim0, size_t dim1, const std::string &name = "transpose");

  static constexpr const char *TYPE_NAME = "transpose";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::shared_ptr<TransposeLayerImpl> create_from_config(const LayerConfig &config);
};

class TransposeLayer : public LayerRef<TransposeLayerImpl> {
public:
  explicit TransposeLayer(size_t dim0, size_t dim1, const std::string &name = "transpose")
      : LayerRef(std::make_shared<TransposeLayerImpl>(dim0, dim1, name)) {}

  using LayerRef<TransposeLayerImpl>::LayerRef;
};

}  // namespace tunx
