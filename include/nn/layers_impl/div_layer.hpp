/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/graph.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class DivLayerImpl : public LayerImpl {
protected:
  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override;
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "div";

  explicit DivLayerImpl(const std::string &name = "div") { this->name_ = name; }

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  Vec<ParamDescriptor> param_descriptors() override { return {}; }

  Node operator()(const Node &a, const Node &b) {
    if (!a || !b) {
      throw std::runtime_error("DivLayerImpl: input nodes cannot be null");
    }
    Graph *graph = a->graph();
    if (graph != b->graph()) {
      throw std::runtime_error("DivLayerImpl: both input nodes must belong to the same graph");
    }
    Node output = graph->make_node();
    std::shared_ptr<LayerImpl> self = shared_from_this();
    graph->add_edge(self, {a, b}, {output});
    return output;
  }

  static std::shared_ptr<DivLayerImpl> create_from_config(const LayerConfig &config);
};

class DivLayer : public LayerRef<DivLayerImpl> {
public:
  explicit DivLayer(const std::string &name = "div")
      : LayerRef(std::make_shared<DivLayerImpl>(name)) {}

  using LayerRef<DivLayerImpl>::LayerRef;
};

}  // namespace tunx
