#pragma once

#include <cstddef>

#include "device/del_allocator_v2.hpp"
#include "nn/graph.hpp"
#include "type/type.hpp"

namespace tnn {

using InputPack = std::unordered_map<std::string, Tensor*>;
using OutputPack = std::unordered_map<std::string, Tensor*>;

class GraphExecutor {
public:
  GraphExecutor(Graph& graph, std::shared_ptr<DELAllocatorV2> ws_allocator)
      : graph_(graph),
        ws_allocator_(ws_allocator) {
    // initialize node outputs
    for (auto& [uid, node] : graph_.io_nodes()) {
      node_outputs_[&node] = Output{nullptr, nullptr};
    }
    for (auto& edge : graph_.edges()) {
      edge.op_node().layer()->set_allocator(*ws_allocator_);
    }
  }

  void forward(const InputPack& inputs, OutputPack& outputs) {
    // set input tensors
    for (const auto& [uid, tensor] : inputs) {
      const IONode& input_node = graph_.io_node(uid);
      if (!(*tensor)) {
        throw std::runtime_error("Null input tensor for uid: " + uid + " while executing graph");
      }
      node_outputs_[&input_node].act = *tensor;
    }
    // assuming topologically sorted edges, execute forward pass
    const auto& edges = graph_.edges();
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      forward(*it);
    }
    // gather output tensors
    for (auto& [uid, tensor] : outputs) {
      const IONode& output_node = graph_.io_node(uid);
      *tensor = node_outputs_[&output_node].act;
      node_outputs_[&output_node].act = nullptr;
    }
  }

  void backward(const InputPack& grads, OutputPack& grad_inputs) {
    // set upstream gradients
    for (const auto& [uid, tensor] : grads) {
      const IONode& output_node = graph_.io_node(uid);
      if (!(*tensor)) {
        throw std::runtime_error("Null gradient tensor for uid: " + uid + " while executing graph");
      }
      node_outputs_[&output_node].grad = *tensor;
    }
    const auto& edges = graph_.edges();
    for (auto it = edges.rbegin(); it != edges.rend(); ++it) {
      backward(*it);
    }
    // gather downstream gradients
    for (auto& [uid, tensor] : grad_inputs) {
      const IONode& input_node = graph_.io_node(uid);
      *tensor = node_outputs_[&input_node].grad;
    }
  }

private:
  Graph& graph_;
  std::shared_ptr<DELAllocatorV2>
      ws_allocator_;  // separate allocator for workspace to allow reuse across layers
  struct Output {
    Tensor act;
    Tensor grad;
  };

  std::unordered_map<const IONode*, Output> node_outputs_;

  void forward(const Edge& edge) {
    Layer* layer = edge.op_node().layer();
    Vec<Vec<size_t>> input_shapes;
    // gather inputs
    const Vec<const IONode*>& input_nodes = edge.producers();
    Vec<ConstTensor> inputs;
    for (const auto& input_node : input_nodes) {
      const ConstTensor& act = node_outputs_[input_node].act;
      if (!act) {
        throw std::runtime_error("Null input while forwarding graph");
      }
      inputs.push_back(act);
      input_shapes.push_back(act->shape());
    }

    // gather outputs
    const Vec<const IONode*>& output_nodes = edge.consumers();

    Vec<Tensor> outputs = layer->forward(inputs);

    for (size_t i = 0; i < output_nodes.size(); ++i) {
      node_outputs_[output_nodes[i]].act = outputs[i];
    }
  }

  void backward(const Edge& edge) {
    Layer* layer = edge.op_node().layer();
    // gather upstream grad_output
    const Vec<const IONode*>& output_nodes = edge.consumers();
    Vec<ConstTensor> gradients;
    for (const auto& output_node : output_nodes) {
      Tensor& grad = node_outputs_[output_node].grad;
      if (!grad) {
        throw std::runtime_error("Null output gradient while backwarding graph");
      }
      gradients.push_back(grad);
    }

    // gather downstream grad_output
    const Vec<const IONode*>& input_nodes = edge.producers();

    Vec<Tensor> grad_inputs = layer->backward(gradients);
    for (size_t i = 0; i < input_nodes.size(); ++i) {
      node_outputs_[input_nodes[i]].grad = grad_inputs[i];
    }
  }
};
}  // namespace tnn