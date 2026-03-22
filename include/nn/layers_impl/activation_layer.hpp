/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nn/activations_impl/base_activation.hpp"
#include "nn/layer.hpp"
#include "stateless_layer.hpp"

namespace tnn {

class ActivationLayer : public StatelessLayer {
private:
  std::unique_ptr<ActivationFunction> activation_;

  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "activation";

  explicit ActivationLayer(std::unique_ptr<ActivationFunction> activation,
                           const std::string &name = "activation");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<ActivationLayer> create_from_config(const LayerConfig &config);
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override {
    return get_shapes_bytes(input_shapes, io_dtype_);
  }
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

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
};

}  // namespace tnn
