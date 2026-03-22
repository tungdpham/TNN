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

#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class SliceLayer : public StatelessLayer {
private:
  std::unordered_map<size_t, Vec<size_t>> micro_batch_original_shapes_;
  size_t axis_;
  size_t start_;
  size_t length_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> slice_forward(const ConstTensor &input, const Tensor &output,
                                      flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> slice_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                       const Vec<size_t> &original_shape,
                                       flowHandle_t handle) const;

  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "slice";

  SliceLayer(size_t axis, size_t start, size_t length, const std::string &name = "slice");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    auto output_shapes = this->output_shapes(input_shapes);
    return get_shapes_bytes(output_shapes, io_dtype_);
  }
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    auto output_shapes = this->output_shapes(input_shapes);
    return get_shapes_bytes(output_shapes, io_dtype_);
  }
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    return get_shapes_bytes(input_shapes, io_dtype_);
  }

  static std::unique_ptr<SliceLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
