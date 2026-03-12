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
  std::unique_ptr<Task> accumulate_pos_gradients(const ConstTensor &grad_output,
                                                 const Tensor &pos_embedding_gradients,
                                                 flowHandle_t handle) const;

  std::vector<ParamDescriptor> param_descriptors() override {
    std::vector<ParamDescriptor> descriptors;
    auto pos_emb_desc = ParamDescriptor{
        param_dtype_,
        {seq_len_, embed_dim_},
        &pos_embedding_,
        &pos_embedding_gradients_,
    };
    descriptors.push_back(pos_emb_desc);
    return descriptors;
  }

  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  explicit PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                                    const std::string &name = "pos_embedding");

  static constexpr const char *TYPE_NAME = "pos_embedding";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

public:
  static std::unique_ptr<PositionalEmbeddingLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
