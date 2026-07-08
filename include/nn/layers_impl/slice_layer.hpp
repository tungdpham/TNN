/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class SliceLayerImpl : public SISOLayerImpl {
private:
  std::unordered_map<size_t, Vec<size_t>> micro_batch_original_shapes_;
  size_t axis_;
  size_t start_;
  size_t length_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> slice_forward(const Tensor &input, Tensor &output,
                                      flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> slice_backward(const Tensor &grad_output, Tensor &grad_input,
                                       const Vec<size_t> &original_shape,
                                       flowHandle_t handle) const;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "slice";

  SliceLayerImpl(size_t axis, size_t start, size_t length, const std::string &name = "slice");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::shared_ptr<SliceLayerImpl> create_from_config(const LayerConfig &config);
};

class SliceLayer : public LayerRef<SliceLayerImpl> {
public:
  SliceLayer(size_t axis, size_t start, size_t length, const std::string &name = "slice")
      : LayerRef(std::make_shared<SliceLayerImpl>(axis, start, length, name)) {}

  using LayerRef<SliceLayerImpl>::LayerRef;
};

}  // namespace tunx
