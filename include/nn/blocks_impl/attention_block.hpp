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

#include "device/task.hpp"
#include "nn/block.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class AttentionBlock : public Block {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;
  bool is_causal_;

  std::unique_ptr<DenseLayer> q_proj_;
  std::unique_ptr<DenseLayer> k_proj_;
  std::unique_ptr<DenseLayer> v_proj_;
  std::unique_ptr<DenseLayer> out_proj_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_attention_forward(const ConstTensor &q, const ConstTensor &k,
                                                  const ConstTensor &v, const Tensor &output,
                                                  size_t batch_size, size_t seq_len,
                                                  flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_attention_backward(const ConstTensor &q, const ConstTensor &k,
                                                   const ConstTensor &v,
                                                   const ConstTensor &d_attn_out, const Tensor &dq,
                                                   const Tensor &dk, const Tensor &dv,
                                                   size_t batch_size, size_t seq_len,
                                                   flowHandle_t handle);

  Vec<Layer *> layers() override {
    return {q_proj_.get(), k_proj_.get(), v_proj_.get(), out_proj_.get()};
  }

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id = 0) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0) override;

public:
  AttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                 const std::string &name = "attention_block");

  static constexpr const char *TYPE_NAME = "attention_block";

  std::string type() const override { return TYPE_NAME; }

  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  static std::unique_ptr<AttentionBlock> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
