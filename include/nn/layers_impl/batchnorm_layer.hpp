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
#include "../optimizers.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

template <typename T = float> class BatchNormLayer : public ParameterizedLayer<T> {
private:
  size_t num_features_;
  T epsilon_;
  T momentum_;
  bool affine_;

  Tensor<T> gamma_;
  Tensor<T> beta_;
  Tensor<T> gamma_gradients_;
  Tensor<T> beta_gradients_;

  Tensor<T> running_mean_;
  Tensor<T> running_var_;

  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_normalized_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_mean_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_var_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_std_;

public:
  explicit BatchNormLayer(size_t num_features, T epsilon = T(1e-5), T momentum = T(0.1),
                          bool affine = true, const std::string &name = "batchnorm");

  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  uint32_t forward_complexity(std::vector<size_t> input_shape) override;
  uint32_t backward_complexity(std::vector<size_t> gradient_shape) override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
  void clear_gradients() override;

public:
  static std::unique_ptr<Layer<T>> create_from_config(const LayerConfig &config);
};

} // namespace tnn

#include "batchnorm_layer.tpp"