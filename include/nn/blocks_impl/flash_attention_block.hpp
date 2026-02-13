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
  void cudnn_backward(const ConstTensor &gradient, const Tensor &grad_input, size_t mb_id);

  std::unordered_map<size_t, cuda::cudnn_flash_attention::feHandle_t *> fe_handle_cache;
#endif
  std::unordered_map<size_t, AttentionStats> stats_cache;
  size_t get_shape_hash(size_t b, size_t h, size_t s, size_t d) const;

  void init_impl() override;
  void on_set_device(const Device &device) override;
  void on_set_flow_handle(flowHandle_t handle) override;
  void on_set_seed(unsigned long long seed) override;
  void on_set_io_dtype(DType_t dtype) override;
  void on_set_param_dtype(DType_t dtype) override;
  void on_set_compute_dtype(DType_t dtype) override;
  void on_set_training(bool training) override;
  // Expects input: [batch_size, seq_len, embed_dim], output: [batch_size, seq_len, embed_dim]
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                      const std::string &name = "flash_attention_block");

  ~FlashAttentionBlock();

  static constexpr const char *TYPE_NAME = "flash_attention_block";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<FlashAttentionBlock> create_from_config(const LayerConfig &config);
  std::vector<Tensor> parameters() override;
  std::vector<Tensor> gradients() override;
};

}  // namespace tnn
