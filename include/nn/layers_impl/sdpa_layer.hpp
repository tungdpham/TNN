/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

// Scaled Dot-Product Attention Layer
// Accepts 3 inputs: Q (B,H,S,D), K (B,H,S,D), V (B,H,S,D)
// Outputs: O (B,H,S,D)
class SDPALayer : public Layer {
private:
  float attn_scale_;  // Attention scale factor (typically 1/sqrt(head_dim))
  bool is_causal_;    // Whether to apply causal masking
  bool is_training_;

  // Cache input shapes and forward pass data for backward
  std::unordered_map<size_t, std::vector<size_t>> micro_batch_q_shapes_;
  std::unordered_map<size_t, ConstTensor> micro_batch_q_cache_;
  std::unordered_map<size_t, ConstTensor> micro_batch_k_cache_;
  std::unordered_map<size_t, ConstTensor> micro_batch_v_cache_;
  std::unordered_map<size_t, Tensor> micro_batch_stats_cache_;

#ifdef USE_CUDNN
  std::unordered_map<size_t, void *> fe_handle_cache_;  // feHandle_t*
  std::unordered_map<size_t, void *> stats_cache_;      // AttentionStats*
#endif

  template <typename IO_T>
  std::unique_ptr<Task> compute_sdpa_forward_impl(const ConstTensor &q, const ConstTensor &k,
                                                  const ConstTensor &v, const Tensor &output,
                                                  size_t batch_size, size_t num_heads,
                                                  size_t seq_len, size_t head_dim,
                                                  flowHandle_t handle, size_t mb_id) const;

  template <typename IO_T>
  std::unique_ptr<Task> compute_sdpa_backward_impl(
      const ConstTensor &q, const ConstTensor &k, const ConstTensor &v, const ConstTensor &output,
      const ConstTensor &grad_output, const Tensor &grad_q, const Tensor &grad_k,
      const Tensor &grad_v, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
      flowHandle_t handle, size_t mb_id) const;

#ifdef USE_CUDNN
  void cudnn_forward(const ConstTensor &q, const ConstTensor &k, const ConstTensor &v,
                     const Tensor &output, size_t mb_id);
  void cudnn_backward(const ConstTensor &q, const ConstTensor &k, const ConstTensor &v,
                      const ConstTensor &output, const ConstTensor &grad_output,
                      const Tensor &grad_q, const Tensor &grad_k, const Tensor &grad_v,
                      size_t mb_id);
#endif

  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id = 0) override;
  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "sdpa";

  // attn_scale typically = 1/sqrt(head_dim)
  SDPALayer(float attn_scale = 1.0f, bool is_causal = false, const std::string &name = "sdpa");

  ~SDPALayer() override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  std::vector<ParamDescriptor> param_descriptors() override { return {}; }
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;

  static std::unique_ptr<SDPALayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
