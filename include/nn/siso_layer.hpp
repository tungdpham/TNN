#pragma once

#include "nn/graph.hpp"
#include "nn/layer.hpp"

namespace tunx {
class SISOLayerImpl : public LayerImpl {
public:
  SISOLayerImpl(const std::string &name)
      : LayerImpl(name) {}

  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override {
    if (inputs.size() != 1) {
      throw std::runtime_error("SISOLayerImpl only supports single input");
    }
    return {forward_impl(inputs[0], residuals)};
  }
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override {
    if (grad_outputs.size() != 1) {
      throw std::runtime_error("SISOLayerImpl only supports single grad output");
    }
    return {backward_impl(grad_outputs[0], residuals)};
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

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

protected:
  virtual Tensor forward_impl(const Tensor &input, Residuals &residuals) = 0;
  virtual Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) = 0;

  virtual Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const = 0;
};

}  // namespace tunx