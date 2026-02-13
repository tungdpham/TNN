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
#include <vector>

#include "device/task.hpp"
#include "stateless_layer.hpp"

namespace tnn {

class LegacyMaxPool2DLayer : public StatelessLayer {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  std::unordered_map<size_t, Tensor> micro_batch_mask_indices_;
  std::unordered_map<size_t, std::vector<size_t>> micro_batch_input_shapes_;

  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> backward_task_;

  template <typename Compute_T>
  std::unique_ptr<Task> compute_max_pool_forward_impl(const ConstTensor &input_data,
                                                      const Tensor &output_data, size_t batch_size,
                                                      size_t channels, size_t input_h,
                                                      size_t input_w, size_t output_h,
                                                      size_t output_w, const Tensor &mask_indices,
                                                      flowHandle_t handle) const;

  std::unique_ptr<Task> compute_max_pool_forward(const ConstTensor &input_data,
                                                 const Tensor &output_data, size_t batch_size,
                                                 size_t channels, size_t input_h, size_t input_w,
                                                 size_t output_h, size_t output_w,
                                                 const Tensor &mask_indices,
                                                 flowHandle_t handle) const;

  template <typename Compute_T>
  std::unique_ptr<Task> compute_max_pool_backward_impl(const ConstTensor &gradient_data,
                                                       const Tensor &grad_input_data,
                                                       size_t batch_size, size_t channels,
                                                       size_t output_h, size_t output_w,
                                                       const ConstTensor &mask_indices,
                                                       flowHandle_t handle) const;

  std::unique_ptr<Task> compute_max_pool_backward(const ConstTensor &gradient_data,
                                                  const Tensor &grad_input_data, size_t batch_size,
                                                  size_t channels, size_t output_h, size_t output_w,
                                                  const ConstTensor &mask_indices,
                                                  flowHandle_t handle) const;

  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  LegacyMaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                       size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "maxpool2d");

  static constexpr const char *TYPE_NAME = "legacy_maxpool2d";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<LegacyMaxPool2DLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
