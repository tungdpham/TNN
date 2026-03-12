#pragma once

#include <cstddef>

#include "device/del_allocator.hpp"
#include "device/iallocator.hpp"
#include "nn/graph.hpp"
#include "type/type.hpp"

namespace tnn {

using InputPack = std::unordered_map<std::string, Tensor>;
using OutputPack = std::unordered_map<std::string, Tensor>;

class GraphExecutor {
public:
  GraphExecutor(Graph& graph, IAllocator& allocator)
      : graph_(graph),
        allocator_(allocator) {
    // initialize node outputs
    for (auto& [uid, node] : graph_.io_nodes()) {
      node_outputs_[&node] = Output{nullptr, nullptr};
    }
    ws_allocator_ = DELAllocator::create(graph_.device(), defaultFlowHandle);
    for (auto& op : graph_.ops()) {
      op->layer()->set_allocator(*ws_allocator_);
    }
  }

  void forward(const InputPack& inputs, OutputPack& outputs) {
    // set input tensors
    for (const auto& [uid, tensor] : inputs) {
      const IONode& input_node = graph_.io_node(uid);
      if (!tensor) {
        throw std::runtime_error("Null input tensor for uid: " + uid + " while executing graph");
      }
      node_outputs_[&input_node].act = tensor;
    }
    // set output tensors
    for (auto& [uid, tensor] : outputs) {
      const IONode& output_node = graph_.io_node(uid);
      if (!tensor) {
        throw std::runtime_error("Null output tensor for uid: " + uid + " while executing graph");
      }
      node_outputs_[&output_node].act = tensor;
    }
    // assuming topologically sorted layers, execute forward pass
    const auto& ops = graph_.ops();
    for (OpNode* op : ops) {
      forward(*op);
    }
  }

  void backward(const InputPack& grads, OutputPack& grad_inputs) {
    // set upstream gradients
    for (const auto& [uid, tensor] : grads) {
      const IONode& output_node = graph_.io_node(uid);
      if (!tensor) {
        throw std::runtime_error("Null gradient tensor for uid: " + uid + " while executing graph");
      }
      node_outputs_[&output_node].grad = tensor;
    }
    // set downstream gradients
    for (auto& [uid, tensor] : grad_inputs) {
      const IONode& input_node = graph_.io_node(uid);
      if (!tensor) {
        throw std::runtime_error("Null gradient tensor for uid: " + uid + " while executing graph");
      }
      node_outputs_[&input_node].grad = tensor;
    }
    const auto& ops = graph_.ops();
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      backward(**it);
    }
  }

private:
  Graph& graph_;
  IAllocator& allocator_;                       // long lived tensors
  std::shared_ptr<DELAllocator> ws_allocator_;  // workspace allocator. short lived tensors

  struct Output {
    Tensor act;
    Tensor grad;
  };

  std::unordered_map<const IONode*, Output> node_outputs_;

  void forward(OpNode& node) {
    Vec<Vec<size_t>> input_shapes;
    // gather inputs
    const std::vector<IONode*>& input_nodes = node.inputs();
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
    const std::vector<IONode*>& output_nodes = node.outputs();
    Vec<Tensor> outputs;
    auto out_shapes = node.output_shapes(input_shapes);
    DType_t dtype = inputs[0]->data_type();
    for (size_t i = 0; i < output_nodes.size(); ++i) {
      IONode* output_node = output_nodes[i];
      Vec<size_t> out_shape = out_shapes[i];
      Tensor& act = node_outputs_[output_node].act;
      if (!act) {
        act = make_tensor(allocator_, dtype, out_shape);
      } else {
        act->ensure(out_shape);
      }
      outputs.push_back(act);
    }

    size_t ws_bytes = node.layer()->is_training()
                          ? std::max(node.layer()->fwd_workspace(input_shapes),
                                     node.layer()->bwd_workspace(input_shapes))
                          : node.layer()->inf_workspace(input_shapes);
    ws_allocator_->reserve(ws_bytes);
    node.forward(inputs, outputs);
  }

  void backward(OpNode& node) {
    // gather upstream grad_output
    const std::vector<IONode*>& output_nodes = node.outputs();
    Vec<ConstTensor> gradients;
    for (const auto& output_node : output_nodes) {
      Tensor& grad = node_outputs_[output_node].grad;
      if (!grad) {
        throw std::runtime_error("Null output gradient while backwarding graph");
      }
      gradients.push_back(grad);
    }

    // gather downstream grad_output
    Vec<Vec<size_t>> input_shapes;
    const std::vector<IONode*>& input_nodes = node.inputs();
    Vec<Tensor> grad_inputs;
    for (const auto& input_node : input_nodes) {
      Tensor& grad = node_outputs_[input_node].grad;
      // just use the same shape and dtype as input activation, since we don't have the actual
      // input tensor here
      auto& input_act = node_outputs_[input_node].act;
      if (!input_act) {
        throw std::runtime_error("Null input activation while backwarding graph");
      }
      if (!grad) {
        grad = make_tensor(allocator_, input_act->data_type(), input_act->shape());
      } else {
        grad->ensure(input_act->shape());
      }
      grad_inputs.push_back(grad);
      input_shapes.push_back(input_act->shape());
    }
    node.backward(gradients, grad_inputs);
  }
};
}  // namespace tnn