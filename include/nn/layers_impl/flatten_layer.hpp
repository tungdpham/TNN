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

#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class FlattenLayer : public StatelessLayer {
private:
  std::unordered_map<size_t, Vec<size_t>> micro_batch_original_shapes_;
  int start_dim_;
  int end_dim_;

  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  explicit FlattenLayer(int start_dim = 1, int end_dim = -1, const std::string &name = "flatten");

  static constexpr const char *TYPE_NAME = "flatten";

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
  static std::unique_ptr<FlattenLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
