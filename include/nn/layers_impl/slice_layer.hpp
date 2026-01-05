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
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class SliceLayer : public StatelessLayer<T> {
private:
  std::unordered_map<size_t, std::vector<size_t>> micro_batch_original_shapes_;
  size_t axis_;
  size_t start_;
  size_t length_;

public:
  SliceLayer(size_t axis, size_t start, size_t length, const std::string &name = "slice");

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override;
  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override;

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

#include "nn/layers_impl/slice_layer.tpp"
