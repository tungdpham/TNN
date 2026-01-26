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

#include "nn/activations_impl/base_activation.hpp"
#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class ActivationLayer : public StatelessLayer {
private:
  std::unique_ptr<ActivationFunction> activation_;
  std::unordered_map<size_t, Tensor> micro_batch_inputs_;

  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "activation";

  explicit ActivationLayer(std::unique_ptr<ActivationFunction> activation,
                           const std::string &name = "activation");

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<ActivationLayer> create_from_config(const LayerConfig &config);
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  size_t cached_memory_bytes() const override;
};

} // namespace tnn
