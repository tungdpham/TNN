/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/block.hpp"
#include "nn/blocks_impl/common/flash_attention.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#ifdef USE_CUDNN
#include "device/task.hpp"
#include "nn/blocks_impl/cuda/cudnn_flash_attention_ops.hpp"
#endif
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace synet {

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

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> flash_attention_forward_task(
      cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
      const ConstTensor &q_heads, const ConstTensor &k_heads, const ConstTensor &v_heads,
      const Tensor &attn_heads, const Tensor &stats_tensor, const Tensor &workspace,
      flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> flash_attention_backward_task(
      cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
      const ConstTensor &q_heads, const ConstTensor &k_heads, const ConstTensor &v_heads,
      const ConstTensor &attn_heads, const ConstTensor &grad_attn_heads,
      const ConstTensor &stats_tensor, const Tensor &grad_q_heads, const Tensor &grad_k_heads,
      const Tensor &grad_v_heads, const Tensor &workspace, flowHandle_t handle) const;

  Tensor cudnn_forward(const ConstTensor &input, size_t mb_id);
  Tensor cudnn_backward(const ConstTensor &grad_output, size_t mb_id);

  mutable std::unordered_map<size_t, cuda::cudnn_flash_attention::feHandle_t *> fe_handle_cache;
#endif
  mutable std::unordered_map<size_t, AttentionStats> stats_cache;

  Vec<LayerImpl *> layers() override {
    return {q_proj_.get(), k_proj_.get(), v_proj_.get(), out_proj_.get()};
  }

  // Expects input: [batch_size, seq_len, embed_dim], output: [batch_size, seq_len, embed_dim]
  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id = 0) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0) override;

public:
  FlashAttentionBlockImpl(size_t embed_dim, size_t num_heads, bool is_causal = true,
                          const std::string &name = "flash_attention_block");

  ~FlashAttentionBlockImpl();

  static constexpr const char *TYPE_NAME = "flash_attention_block";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  static std::shared_ptr<FlashAttentionBlockImpl> create_from_config(const LayerConfig &config);

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

}  // namespace synet
