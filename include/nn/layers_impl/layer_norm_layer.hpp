/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <string>

namespace tnn {

template <typename T = float> class LayerNormLayer : public ParameterizedLayer<T> {
private:
  size_t normalized_shape_; // Size of C (channels)
  T epsilon_;
  bool affine_; // Whether to use learnable affine parameters

  Tensor<T> gamma_;
  Tensor<T> beta_;
  Tensor<T> gamma_gradients_;
  Tensor<T> beta_gradients_;
  std::map<size_t, Tensor<T>> micro_batch_inputs_;

public:
  explicit LayerNormLayer(size_t normalized_shape, T epsilon = 1e-5, bool affine = true,
                          const std::string &name = "layer_norm");

  void initialize_params() override;

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override;
  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override;

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return "layer_norm"; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape;
  }

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
};

} // namespace tnn

#include "nn/layers_impl/layer_norm_layer.tpp"
