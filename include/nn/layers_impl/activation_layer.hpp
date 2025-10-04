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

#include "../activations.hpp"
#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

template <typename T = float> class ActivationLayer : public StatelessLayer<T> {
private:
  std::unique_ptr<ActivationFunction<T>> activation_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;

public:
  explicit ActivationLayer(std::unique_ptr<ActivationFunction<T>> activation,
                           const std::string &name = "activation");

  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  void forward_inplace(Tensor<T> &input, size_t micro_batch_id = 0) override;
  void backward_inplace(Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) override;
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t forward_memory_traffic(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_memory_traffic(const std::vector<size_t> &input_shape) const override;
  double forward_arithmetic_intensity(const std::vector<size_t> &input_shape) const override;
  double backward_arithmetic_intensity(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
};

} // namespace tnn