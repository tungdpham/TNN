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
#include <vector>

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

  void init_impl() override;
  void on_set_device(const Device &device) override;
  void on_set_flow_handle(flowHandle_t handle) override;
  void on_set_seed(unsigned long long seed) override;
  void on_set_io_dtype(DType_t dtype) override;
  void on_set_param_dtype(DType_t dtype) override;
  void on_set_compute_dtype(DType_t dtype) override;
  void on_set_training(bool training) override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  AttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                 const std::string &name = "attention_block");

  static constexpr const char *TYPE_NAME = "attention_block";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<AttentionBlock> create_from_config(const LayerConfig &config);

  std::vector<Tensor> parameters() override;
  std::vector<Tensor> gradients() override;
};

}  // namespace tnn
