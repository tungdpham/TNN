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

#include "nn/layers_impl/stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class TransposeLayer : public StatelessLayer {
private:
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> permute(const ConstTensor &input, Tensor &output, size_t B, size_t L,
                                size_t H, size_t D, const std::string &flow_id) const;

  void forward_impl(const ConstTensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;

public:
  TransposeLayer(const std::string &name = "transpose");

  static constexpr const char *TYPE_NAME = "transpose";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override { return 0; }

  static std::unique_ptr<TransposeLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
