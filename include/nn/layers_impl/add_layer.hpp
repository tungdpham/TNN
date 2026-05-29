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
class AddLayerImpl : public LayerImpl {
protected:
  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) override;

public:
  static constexpr const char *TYPE_NAME = "add";

  explicit AddLayerImpl(const std::string &name = "add") { this->name_ = name; }

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  Vec<ParamDescriptor> param_descriptors() override { return {}; }

  Node operator()(const Node &a, const Node &b) {
    if (!a || !b) {
      throw std::runtime_error("AddLayerImpl: input nodes cannot be null");
    }
    Graph *graph = a->graph();
    if (graph != b->graph()) {
      throw std::runtime_error("AddLayerImpl: both input nodes must belong to the same graph");
    }
    Node output = graph->make_node();
    std::shared_ptr<LayerImpl> self = shared_from_this();
    graph->add_edge(self, {a, b}, {output});
    return output;
  }

  static std::shared_ptr<AddLayerImpl> create_from_config(const LayerConfig &config);
};

class AddLayer : public LayerRef<AddLayerImpl> {
public:
  AddLayer(const std::string &name = "add")
      : LayerRef(std::make_shared<AddLayerImpl>(name)) {}

  using LayerRef<AddLayerImpl>::LayerRef;
};

}  // namespace tnn
