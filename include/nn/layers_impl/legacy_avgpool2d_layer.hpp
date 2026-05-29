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
#include <unordered_map>

#include "device/task.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class LegacyAvgPool2DLayerImpl : public SISOLayerImpl {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  // Cache input shapes for backward pass
  std::unordered_map<size_t, Vec<size_t>> micro_batch_input_shapes_;

  template <typename Compute_T>
  std::unique_ptr<Task> run_forward(const ConstTensor &input_data, const Tensor &output_data,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w,
                                    flowHandle_t handle) const;

  std::unique_ptr<Task> run_forward(const ConstTensor &input_data, const Tensor &output_data,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w,
                                    flowHandle_t handle) const;

  template <typename Compute_T>
  std::unique_ptr<Task> run_backward(const ConstTensor &gradient_data,
                                     const Tensor &grad_input_data, size_t batch_size,
                                     size_t channels, size_t input_h, size_t input_w,
                                     size_t output_h, size_t output_w, flowHandle_t handle) const;

  std::unique_ptr<Task> run_backward(const ConstTensor &gradient_data,
                                     const Tensor &grad_input_data, size_t batch_size,
                                     size_t channels, size_t input_h, size_t input_w,
                                     size_t output_h, size_t output_w, flowHandle_t handle) const;

  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  LegacyAvgPool2DLayerImpl(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                           size_t pad_h = 0, size_t pad_w = 0,
                           const std::string &name = "avgpool2d");

  static constexpr const char *TYPE_NAME = "legacy_avgpool2d";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<LegacyAvgPool2DLayerImpl> create_from_config(const LayerConfig &config);
};

class LegacyAvgPool2DLayer : public LayerRef<LegacyAvgPool2DLayerImpl> {
public:
  LegacyAvgPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                       size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "avgpool2d")
      : LayerRef(std::make_shared<LegacyAvgPool2DLayerImpl>(pool_h, pool_w, stride_h, stride_w,
                                                            pad_h, pad_w, name)) {}

  using LayerRef<LegacyAvgPool2DLayerImpl>::LayerRef;
};

}  // namespace tnn
