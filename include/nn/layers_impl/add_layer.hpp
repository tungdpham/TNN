/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/graph_api.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

// Element-wise addition layer.
class AddLayer : public Layer {
protected:
  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) override;

public:
  static constexpr const char *TYPE_NAME = "add";

  explicit AddLayer(const std::string &name = "add") { this->name_ = name; }

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  Vec<ParamDescriptor> param_descriptors() override { return {}; }

  graph_api_v2::Node operator()(const graph_api_v2::Node &a, const graph_api_v2::Node &b) {
    if (!a || !b) {
      throw std::runtime_error("AddLayer: input nodes cannot be null");
    }
    graph_api_v2::Graph *graph = a->graph();
    if (graph != b->graph()) {
      throw std::runtime_error("AddLayer: both input nodes must belong to the same graph");
    }
    graph_api_v2::Node output = graph->make_node();
    std::shared_ptr<Layer> self = shared_from_this();
    graph->add_edge(self, {a, b}, {output});
    return output;
  }

  static std::unique_ptr<AddLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
