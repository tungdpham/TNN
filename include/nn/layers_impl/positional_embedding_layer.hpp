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

#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class PositionalEmbeddingLayer : public ParameterizedLayer {
private:
  size_t embed_dim_;
  size_t seq_len_;
  Tensor pos_embedding_;
  Tensor pos_embedding_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_positional_embedding(const ConstTensor &input, const Tensor &output,
                                                 const ConstTensor &pos_embedding,
                                                 flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> accumulate_pos_gradients(const ConstTensor &gradient,
                                                 const Tensor &pos_embedding_gradients,
                                                 flowHandle_t handle) const;

  void init_params() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                     size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  explicit PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                                    const std::string &name = "pos_embedding");

  static constexpr const char *TYPE_NAME = "pos_embedding";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }

  LayerConfig get_config() const override;

  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

public:
  static std::unique_ptr<PositionalEmbeddingLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
