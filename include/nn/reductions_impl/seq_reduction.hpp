#pragma once

#include <memory>
#include <stack>
#include <unordered_map>

#include "nn/blocks_impl/sequential.hpp"
#include "nn/graph_builder.hpp"
#include "nn/io_node.hpp"
#include "nn/op_node.hpp"
#include "nn/reduction_strategy.hpp"

namespace tnn {
class SeqReduction : public ReductionStrategy {
public:
  void reduce(GraphBuilder &builder) override {
    std::unordered_map<std::string, OpNode> &op_nodes = builder.op_nodes();
    std::vector<OpNode *> &execution_sequence = builder.execution_sequence();
    std::stack<std::string> node_stack;
    std::vector<std::string> op_to_erase;

    auto reduce_func = [&]() {
      if (node_stack.size() <= 1) {
        while (!node_stack.empty()) {
          node_stack.pop();
        }
        return;
      }
      Vec<std::unique_ptr<Layer>> layers;
      IONode *pre = nullptr;   // io node before the sequence of op nodes
      IONode *next = nullptr;  // io node after the sequence of op nodes
      while (!node_stack.empty()) {
        auto top_uid = node_stack.top();
        auto it = op_nodes.find(top_uid);
        if (it == op_nodes.end()) {
          throw std::runtime_error("Op node not found in graph builder?");
        }
        OpNode top_node = std::move(it->second);
        op_to_erase.push_back(top_uid);
        node_stack.pop();
        pre = top_node.inputs()[0];
        if (!next) {
          next = top_node.outputs()[0];
        }
        std::unique_ptr<Layer> layer = top_node.release_layer();
        layers.push_back(std::move(layer));
      }
      std::reverse(layers.begin(), layers.end());
      std::unique_ptr<Sequential> sequential_layer =
          std::make_unique<Sequential>(std::move(layers));
      sequential_layer->print_summary({1, 32, 32, 3});
      OpNode &new_op_node = builder.add_layer(std::move(sequential_layer));
      pre->clear();
      next->clear();
      pre->add_consumer(&new_op_node);
      next->add_producer(&new_op_node);
      new_op_node.add_input(pre);
      new_op_node.add_output(next);
    };

    for (const auto &op_node : execution_sequence) {
      // only nodes with linear input, output and consumer
      if (op_node->inputs().size() == 1 && op_node->outputs().size() == 1 &&
          op_node->outputs()[0]->consumers().size() <= 1) {
        node_stack.push(op_node->uid());
      } else {
        reduce_func();
      }
    }

    reduce_func();

    for (const auto &uid : op_to_erase) {
      op_nodes.erase(uid);
    }

    builder.sort();
  }
};
}  // namespace tnn