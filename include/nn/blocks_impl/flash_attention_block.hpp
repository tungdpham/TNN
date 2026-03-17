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
#include <vector>

namespace tnn {

class FlashAttentionBlock : public Block {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;
  bool is_causal_;

  std::unique_ptr<DenseLayer> q_proj_;
  std::unique_ptr<DenseLayer> k_proj_;
  std::unique_ptr<DenseLayer> v_proj_;
  std::unique_ptr<DenseLayer> out_proj_;

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

  void cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id);
  void cudnn_backward(const ConstTensor &grad_output, const Tensor &grad_input, size_t mb_id);

  mutable std::unordered_map<size_t, cuda::cudnn_flash_attention::feHandle_t *> fe_handle_cache;
#endif
  mutable std::unordered_map<size_t, AttentionStats> stats_cache;

  std::vector<Layer *> layers() override {
    return {q_proj_.get(), k_proj_.get(), v_proj_.get(), out_proj_.get()};
  }

  // Expects input: [batch_size, seq_len, embed_dim], output: [batch_size, seq_len, embed_dim]
  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id = 0) override;
  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id = 0) override;

public:
  FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                      const std::string &name = "flash_attention_block");

  ~FlashAttentionBlock();

  static constexpr const char *TYPE_NAME = "flash_attention_block";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  static std::unique_ptr<FlashAttentionBlock> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
