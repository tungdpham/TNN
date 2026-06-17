/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#include "nn/block.hpp"
#include "nn/graph.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace synet {
class SequentialImpl : public Block {
private:
  Vec<Layer> layers_;

protected:
  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override;
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override;

public:
  explicit SequentialImpl(Vec<Layer> layers = {}, const std::string &name = "sequential");

  explicit SequentialImpl(std::initializer_list<Layer> layers,
                          const std::string &name = "sequential")
      : SequentialImpl(Vec<Layer>(layers), name) {}

  explicit SequentialImpl(Vec<std::shared_ptr<LayerImpl>> layers,
                          const std::string &name = "sequential")
      : SequentialImpl(Vec<Layer>(layers.begin(), layers.end()), name) {}

  static constexpr const char *TYPE_NAME = "sequential";

  std::string type() const override { return TYPE_NAME; }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  void print_summary(const Vec<size_t> &input_shape) const;
  LayerConfig get_config() const override;
  static std::shared_ptr<SequentialImpl> create_from_config(const LayerConfig &config);

  Vec<Layer> layers() override { return layers_; }

  Node operator()(const Node &input) {
    if (!input) {
      throw std::runtime_error("Input node is null");
    }
    Graph *graph = input->graph();
    Node output = graph->make_node();

    std::shared_ptr<LayerImpl> self = shared_from_this();

    graph->add_edge(self, {input}, {output});
    return output;
  }
};

class Sequential : public LayerRef<SequentialImpl> {
public:
  explicit Sequential(Vec<Layer> layers, const std::string &name = "sequential")
      : LayerRef(std::make_shared<SequentialImpl>(std::move(layers), name)) {}

  explicit Sequential(std::initializer_list<Layer> layers, const std::string &name = "sequential")
      : Sequential(Vec<Layer>(layers), name) {}

  Vec<Layer> layers() { return impl_->layers(); }

  void print_summary(const Vec<size_t> &input_shape) const { impl_->print_summary(input_shape); }

  using LayerRef<SequentialImpl>::LayerRef;
};

}  // namespace synet