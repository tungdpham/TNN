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

class PositionalEmbeddingLayer : public ParameterizedLayer {
private:
  size_t embed_dim_;
  size_t seq_len_;
  Tensor pos_embedding_;
  Tensor pos_embedding_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_positional_embedding(const Tensor &input, Tensor &output,
                                                 const Tensor &pos_embedding,
                                                 const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> accumulate_pos_gradients(const Tensor &gradient,
                                                 Tensor &pos_embedding_gradients,
                                                 const std::string &flow_id) const;

  void init_params() override;
  void forward_impl(const Tensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  static constexpr const char *TYPE_NAME = "pos_embedding";

  explicit PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                                    const std::string &name = "pos_embedding");

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }

  LayerConfig get_config() const override;

  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

public:
  static std::unique_ptr<PositionalEmbeddingLayer> create_from_config(const LayerConfig &config);
};

} // namespace tnn
