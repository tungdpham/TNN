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

class FlattenLayer : public StatelessLayer {
private:
  std::unordered_map<size_t, std::vector<size_t>> micro_batch_original_shapes_;
  int start_dim_;
  int end_dim_;

  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  explicit FlattenLayer(int start_dim = 1, int end_dim = -1, const std::string &name = "flatten");

  static constexpr const char *TYPE_NAME = "flatten";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<FlattenLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
