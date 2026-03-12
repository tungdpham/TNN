#pragma once

#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/graph.hpp"
#include "nn/io_node.hpp"
#include "nn/op_node.hpp"

namespace tnn {

class GraphBuilder {
public:
  GraphBuilder()
      : ctx_desc_{
            .param_descs = {},
            .param_bytes = 0,
            .grad_bytes = 0,
        } {}

  size_t num_nodes() const { return op_nodes_.size() + io_nodes_.size(); }

  OpNode& add_layer(std::unique_ptr<Layer> siso_layer) {
    std::string uid = "op_" + std::to_string(node_count_++);
    OpNode new_node(uid, ctx_desc_, std::move(siso_layer));
    auto& node = add_op_node(std::move(new_node));
    return node;
  }

  IONode& input(std::string uid = "input") {
    auto node = IONode(uid);
    auto& input_node = add_io_node(std::move(node));
    return input_node;
  }

  // in reality, only the specific layer knows how many outputs it has, but for simplicity we assume
  // it's always 1-1
  IONode& output(OpNode& op_node, IONode& input, std::string uid = "") {
    uid = uid.empty() ? "io_" + std::to_string(node_count_++) : uid;
    auto new_node = IONode(uid);
    auto& output = add_io_node(std::move(new_node));
    add_out(op_node, output);
    add_in(op_node, input);
    return output;
  }

  void sort() {
    execution_sequence_.clear();
    if (op_nodes_.empty()) return;

    std::unordered_map<OpNode*, int> in_degree;
    for (auto& op_pair : op_nodes_) {
      OpNode& op = op_pair.second;
      in_degree[&op] = 0;
      for (IONode* in_tensor : op.inputs()) {
        in_degree[&op] += in_tensor->producers().size();
      }
    }

    std::deque<OpNode*> queue;
    for (auto& op_pair : op_nodes_) {
      OpNode& op = op_pair.second;
      if (in_degree[&op] == 0) {
        queue.push_back(&op);
      }
    }

    while (!queue.empty()) {
      OpNode* curr = queue.front();
      queue.pop_front();
      execution_sequence_.push_back(curr);

      for (IONode* out_tensor : curr->outputs()) {
        for (OpNode* consumer : out_tensor->consumers()) {
          in_degree[consumer]--;
          if (in_degree[consumer] == 0) {
            queue.push_back(consumer);
          }
        }
      }
    }

    if (execution_sequence_.size() != op_nodes_.size()) {
      throw std::runtime_error("Graph contains a cycle");
    }
  }

  std::unordered_map<std::string, IONode>& io_nodes() { return io_nodes_; }
  std::unordered_map<std::string, OpNode>& op_nodes() { return op_nodes_; }
  std::vector<OpNode*>& execution_sequence() {
    if (execution_sequence_.empty()) {
      sort();
    }
    return execution_sequence_;
  }

  Graph compile(IAllocator& allocator) {
    sort();
    return Graph(allocator, ctx_desc_, std::move(op_nodes_), std::move(io_nodes_),
                 std::move(execution_sequence_));
  }

private:
  GraphContextDescriptor ctx_desc_;
  std::unordered_map<std::string, IONode> io_nodes_;
  std::unordered_map<std::string, OpNode> op_nodes_;
  std::vector<OpNode*> execution_sequence_;
  size_t node_count_ = 0;

  void add_in(OpNode& op_node, IONode& input) {
    op_node.add_input(&input);
    input.add_consumer(&op_node);
  }

  void add_out(OpNode& op_node, IONode& output) {
    op_node.add_output(&output);
    output.add_producer(&op_node);
  }

  OpNode& add_op_node(OpNode&& op_node) {
    auto [iter, inserted] = op_nodes_.emplace(op_node.uid(), std::move(op_node));
    if (!inserted) {
      throw std::runtime_error("Duplicate OpNode UID: " + op_node.uid());
    }
    return iter->second;
  }

  IONode& add_io_node(IONode&& io_node) {
    auto [iter, inserted] = io_nodes_.emplace(io_node.uid(), std::move(io_node));
    if (!inserted) {
      throw std::runtime_error("Duplicate IONode UID: " + io_node.uid());
    }
    return iter->second;
  }
};

}  // namespace tnn