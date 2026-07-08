/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class AvgPool2DLayerImpl : public SISOLayerImpl {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "avgpool2d";

  AvgPool2DLayerImpl(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                     size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "avgpool2d");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<AvgPool2DLayerImpl> create_from_config(const LayerConfig &config);
};

class AvgPool2DLayer : public LayerRef<AvgPool2DLayerImpl> {
public:
  AvgPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                 size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "avgpool2d")
      : LayerRef(std::make_shared<AvgPool2DLayerImpl>(pool_h, pool_w, stride_h, stride_w, pad_h,
                                                      pad_w, name)) {}

  using LayerRef<AvgPool2DLayerImpl>::LayerRef;
};

}  // namespace tunx
