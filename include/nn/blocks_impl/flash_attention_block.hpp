/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cmath>
#include <memory>
#include <string>

#include "nn/block.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"

namespace tunx {

class FlashAttentionBlockImpl : public Block {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;
  bool is_causal_;

  DenseLayer q_proj_;
  DenseLayer k_proj_;
  DenseLayer v_proj_;
  DenseLayer out_proj_;

  // Expects input: [batch_size, seq_len, embed_dim], output: [batch_size, seq_len, embed_dim]
  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override;
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override;

public:
  FlashAttentionBlockImpl(size_t embed_dim, size_t num_heads, bool is_causal = true,
                          const std::string &name = "flash_attention_block");

  ~FlashAttentionBlockImpl();

  static constexpr const char *TYPE_NAME = "flash_attention_block";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  static std::shared_ptr<FlashAttentionBlockImpl> create_from_config(const LayerConfig &config);

  Vec<Layer> layers() override { return {q_proj_, k_proj_, v_proj_, out_proj_}; }

  Node operator()(const Node &input) {
    if (!input) {
      throw std::runtime_error("Input node is null");
    }
    Graph *graph = input->graph();
    Node output = graph->make_node();

    std::shared_ptr<LayerImpl> self = shared_from_this();

    graph->add_edge(self, {input}, {output});
    return output;
  }
};

class FlashAttentionBlock : public LayerRef<FlashAttentionBlockImpl> {
public:
  FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                      const std::string &name = "flash_attention_block")
      : LayerRef(std::make_shared<FlashAttentionBlockImpl>(embed_dim, num_heads, is_causal, name)) {
  }

  using LayerRef<FlashAttentionBlockImpl>::LayerRef;
};

}  // namespace tunx
