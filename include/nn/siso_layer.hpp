#pragma once

#include "nn/graph_api.hpp"
#include "nn/layer.hpp"

namespace tnn {
class SISOLayerImpl : virtual public LayerImpl {
public:
  SISOLayerImpl() = default;
  SISOLayerImpl(const std::string &name)
      : LayerImpl(name) {}

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) override {
    if (inputs.size() != 1) {
      throw std::runtime_error("SISOLayerImpl only supports single input");
    }
    return {forward_impl(inputs[0], mb_id)};
  }
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) override {
    if (grad_outputs.size() != 1) {
      throw std::runtime_error("SISOLayerImpl only supports single grad output");
    }
    return {backward_impl(grad_outputs[0], mb_id)};
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

  graph_api_v2::Node operator()(const graph_api_v2::Node &input) {
    if (!input) {
      throw std::runtime_error("Input node is null");
    }
    graph_api_v2::Graph *graph = input->graph();
    graph_api_v2::Node output = graph->make_node();

    std::shared_ptr<LayerImpl> self = shared_from_this();

    graph->add_edge(self, {input}, {output});
    return output;
  }

protected:
  virtual Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) = 0;
  virtual Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) = 0;

  virtual Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const = 0;
};

}  // namespace tnn