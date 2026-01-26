/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace tnn {

class AttentionBlock : public ParameterizedLayer {
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
  std::unique_ptr<Task> compute_attention_forward(const Tensor &q, const Tensor &k, const Tensor &v,
                                                  Tensor &output, size_t batch_size, size_t seq_len,
                                                  const std::string &flow_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task>
  compute_attention_backward(const Tensor &q, const Tensor &k, const Tensor &v,
                             const Tensor &d_attn_out, Tensor &dq, Tensor &dk, Tensor &dv,
                             size_t batch_size, size_t seq_len, const std::string &flow_id);

  void init_params() override;
  void on_set_io_dtype(DType_t dtype) override;
  void on_set_param_dtype(DType_t dtype) override;
  void on_set_device(const Device &device) override;
  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override;

public:
  AttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                 const std::string &name = "attention_block");

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  static constexpr const char *TYPE_NAME = "attention_block";
  std::string type() const override { return TYPE_NAME; }

  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<AttentionBlock> create_from_config(const LayerConfig &config);

protected:
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;
};

} // namespace tnn
