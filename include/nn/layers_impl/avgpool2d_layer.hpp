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
#include <vector>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class AvgPool2DLayer : public StatelessLayer {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  // Cache input shapes for backward pass
  std::unordered_map<size_t, std::vector<size_t>> micro_batch_input_shapes_;

  template <typename Compute_T>
  std::unique_ptr<Task> compute_avg_pool_forward_impl(const ConstTensor &input_data,
                                                      const Tensor &output_data, size_t batch_size,
                                                      size_t height, size_t width, size_t channels,
                                                      size_t output_h, size_t output_w,
                                                      flowHandle_t handle) const;
  template <typename Compute_T>
  std::unique_ptr<Task> compute_avg_pool_backward_impl(const ConstTensor &gradient_data,
                                                       const Tensor &grad_input_data,
                                                       size_t batch_size, size_t input_h,
                                                       size_t input_w, size_t channels,
                                                       size_t output_h, size_t output_w,
                                                       flowHandle_t handle) const;

  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "avgpool2d";

  AvgPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                 size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "avgpool2d");

  // no caching needed
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  // workspace needed to materialize output
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    auto output_shapes = this->output_shapes(input_shapes);
    return get_shapes_bytes(output_shapes, io_dtype_);
  }
  // workspace needed for inference (can be different from forward pass if we can reuse some
  // buffers)
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    auto output_shapes = this->output_shapes(input_shapes);
    return get_shapes_bytes(output_shapes, io_dtype_);
  }
  // workspace needed for backward pass (e.g., can be workspace + input gradient bytes)
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    return get_shapes_bytes(input_shapes, io_dtype_);
  }
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<AvgPool2DLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
