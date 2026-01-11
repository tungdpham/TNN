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
#include <vector>

namespace tnn {

template <typename T = float> class PositionalEmbeddingLayer : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  size_t seq_len_;
  Tensor<T> pos_embedding_;
  Tensor<T> pos_embedding_gradients_;

  void forward_impl(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                     size_t micro_batch_id = 0) override;

public:
  PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                           const std::string &name = "pos_embedding");

  void clear_gradients() override;

  void init_params() override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;

  LayerConfig get_config() const override;

  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
};

} // namespace tnn

#include "nn/layers_impl/positional_embedding_layer.tpp"
