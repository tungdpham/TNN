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

#include "device/task.hpp"
#include "nn/graph_api.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

// Scaled Dot-Product Attention LayerImpl
// Accepts 3 inputs: Q (B,H,S,D), K (B,H,S,D), V (B,H,S,D)
// Outputs: O (B,H,S,D)
class SDPALayerImpl : public LayerImpl {
private:
  float attn_scale_;  // Attention scale factor (typically 1/sqrt(head_dim))
  bool is_causal_;    // Whether to apply causal masking
  bool is_training_;

  // Cache input shapes and forward pass data for backward
  std::unordered_map<size_t, Vec<size_t>> micro_batch_q_shapes_;
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

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id = 0) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0) override;

public:
  static constexpr const char *TYPE_NAME = "sdpa";

  // attn_scale typically = 1/sqrt(head_dim)
  SDPALayerImpl(float attn_scale = 1.0f, bool is_causal = false, const std::string &name = "sdpa");

  ~SDPALayerImpl() override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  Vec<ParamDescriptor> param_descriptors() override { return {}; }

  graph_api_v2::Node operator()(const graph_api_v2::Node &q, const graph_api_v2::Node &k,
                                const graph_api_v2::Node &v) {
    if (!q || !k || !v) {
      throw std::runtime_error("Input nodes cannot be null");
    }
    graph_api_v2::Graph *graph = q->graph();
    if (graph != k->graph() || graph != v->graph()) {
      throw std::runtime_error("All input nodes must belong to the same graph");
    }
    graph_api_v2::Node output = graph->make_node();

    std::shared_ptr<LayerImpl> self = shared_from_this();

    graph->add_edge(self, {q, k, v}, {output});
    return output;
  }

  static std::unique_ptr<SDPALayerImpl> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
