#pragma once

#include "nn/layers_impl/stateless_layer.hpp"

namespace tnn {
class ReLULayer : public StatelessLayer {
  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "activation";

  explicit ReLULayer(const std::string &name = "activation");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<ReLULayer> create_from_config(const LayerConfig &config);
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