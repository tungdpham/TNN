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

#include "nn/graph.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

// Scaled Dot-Product Attention LayerImpl
// Accepts 3 inputs: Q (B,H,S,D), K (B,H,S,D), V (B,H,S,D)
// Outputs: O (B,H,S,D)
class SDPALayerImpl : public LayerImpl {
private:
  float attn_scale_;  // Attention scale factor (typically 1/sqrt(head_dim))
  bool is_causal_;    // Whether to apply causal masking
  bool is_training_;

  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override;
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "sdpa";

  // attn_scale typically = 1/sqrt(head_dim)
  SDPALayerImpl(float attn_scale = 1.0f, bool is_causal = false, const std::string &name = "sdpa");

  ~SDPALayerImpl() override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  Vec<ParamDescriptor> param_descriptors() override { return {}; }

  Node operator()(const Node &q, const Node &k, const Node &v) {
    if (!q || !k || !v) {
      throw std::runtime_error("Input nodes cannot be null");
    }
    Graph *graph = q->graph();
    if (graph != k->graph() || graph != v->graph()) {
      throw std::runtime_error("All input nodes must belong to the same graph");
    }
    Node output = graph->make_node();

    std::shared_ptr<LayerImpl> self = shared_from_this();

    graph->add_edge(self, {q, k, v}, {output});
    return output;
  }

  static std::shared_ptr<SDPALayerImpl> create_from_config(const LayerConfig &config);
};

class SDPALayer : public LayerRef<SDPALayerImpl> {
public:
  explicit SDPALayer(float attn_scale = 1.0f, bool is_causal = false,
                     const std::string &name = "sdpa")
      : LayerRef(std::make_shared<SDPALayerImpl>(attn_scale, is_causal, name)) {}

  using LayerRef<SDPALayerImpl>::LayerRef;
};

}  // namespace tunx
