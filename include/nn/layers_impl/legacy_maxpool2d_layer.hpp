/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "device/task.hpp"
#include "nn/siso_layer.hpp"

namespace synet {

class LegacyMaxPool2DLayerImpl : public SISOLayerImpl {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  template <typename Compute_T>
  std::unique_ptr<Task> run_forward(const Tensor &input_data, Tensor &output_data,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w,
                                    Tensor &mask_indices, flowHandle_t handle) const;

  std::unique_ptr<Task> run_forward(const Tensor &input_data, Tensor &output_data,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w,
                                    Tensor &mask_indices, flowHandle_t handle) const;

  template <typename Compute_T>
  std::unique_ptr<Task> run_backward(const Tensor &gradient_data, Tensor &grad_input_data,
                                     size_t batch_size, size_t channels, size_t output_h,
                                     size_t output_w, const Tensor &mask_indices,
                                     flowHandle_t handle) const;

  std::unique_ptr<Task> run_backward(const Tensor &gradient_data, Tensor &grad_input_data,
                                     size_t batch_size, size_t channels, size_t output_h,
                                     size_t output_w, const Tensor &mask_indices,
                                     flowHandle_t handle) const;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  LegacyMaxPool2DLayerImpl(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                           size_t pad_h = 0, size_t pad_w = 0,
                           const std::string &name = "maxpool2d");

  static constexpr const char *TYPE_NAME = "legacy_maxpool2d";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::shared_ptr<LegacyMaxPool2DLayerImpl> create_from_config(const LayerConfig &config);
};

class LegacyMaxPool2DLayer : public LayerRef<LegacyMaxPool2DLayerImpl> {
public:
  LegacyMaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                       size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "maxpool2d")
      : LayerRef(std::make_shared<LegacyMaxPool2DLayerImpl>(pool_h, pool_w, stride_h, stride_w,
                                                            pad_h, pad_w, name)) {}

  using LayerRef<LegacyMaxPool2DLayerImpl>::LayerRef;
};

}  // namespace synet
