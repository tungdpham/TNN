/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class DropoutLayer : public StatelessLayer<T> {
private:
  T dropout_rate_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_masks_;
  mutable std::mt19937 generator_;

public:
  explicit DropoutLayer(T dropout_rate, const std::string &name = "dropout");

  const Tensor<T> &forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  const Tensor<T> &backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<Layer<T>> create_from_config(const LayerConfig &config);
};

} // namespace tnn

#include "nn/layers_impl/dropout_layer.tpp"