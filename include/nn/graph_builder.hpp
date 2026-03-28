#pragma once

#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
    OpNode new_node(uid, std::move(siso_layer));
    auto& node = add_op_node(std::move(new_node));
    return node;
  }

  IONode& io(std::string uid = "input") {
    auto node = IONode(uid);
    auto& input_node = add_io_node(std::move(node));
    return input_node;
  }

  void add_edge(Vec<const IONode*> producers, Vec<const IONode*> consumers, OpNode& op_node) {
    Edge edge(std::move(producers), std::move(consumers), op_node);
    edges_.push_back(std::move(edge));
  }

  void sort() {
    if (edges_.empty()) return;

    // Kahn's algorithm for topological sorting.
    // An edge A must precede edge B if any of A's consumers appear in B's producers.
    size_t n = edges_.size();

    // Map each IONode to the edge index whose consumers contain it
    std::unordered_map<const IONode*, size_t> produced_by;
    for (size_t i = 0; i < n; ++i) {
      for (const IONode* node : edges_[i].consumers()) {
        produced_by[node] = i;
      }
    }

    // Build adjacency list (A -> B means B depends on A) and in-degree counts
    Vec<Vec<size_t>> adj(n);
    Vec<size_t> in_degree(n, 0);
    for (size_t i = 0; i < n; ++i) {
      for (const IONode* node : edges_[i].producers()) {
        auto it = produced_by.find(node);
        if (it != produced_by.end()) {
          size_t src = it->second;
          adj[src].push_back(i);
          ++in_degree[i];
        }
      }
    }

    // Seed queue with edges that have no unresolved inputs
    std::deque<size_t> queue;
    for (size_t i = 0; i < n; ++i) {
      if (in_degree[i] == 0) queue.push_back(i);
    }

    Vec<Edge> sorted;
    sorted.reserve(n);
    while (!queue.empty()) {
      size_t idx = queue.front();
      queue.pop_front();
      sorted.push_back(std::move(edges_[idx]));
      for (size_t dep : adj[idx]) {
        if (--in_degree[dep] == 0) queue.push_back(dep);
      }
    }

    if (sorted.size() != n) {
      throw std::runtime_error("Cycle detected in graph: topological sort failed");
    }

    edges_ = std::move(sorted);
  }

  std::unordered_map<std::string, IONode>& io_nodes() { return io_nodes_; }
  std::unordered_map<std::string, OpNode>& op_nodes() { return op_nodes_; }
  Vec<Edge>& edges() { return edges_; }

  Graph compile(IAllocator& allocator) {
    sort();
    for (auto& [uid, node] : op_nodes_) {
      node.layer()->set_engine_type(allocator.device().get_engine());
      auto param_descriptors = node.layer()->param_descriptors();
      for (const auto& desc : param_descriptors) {
        ctx_desc_.register_desc(desc);
      }
    }
    return Graph(allocator, ctx_desc_, std::move(op_nodes_), std::move(io_nodes_),
                 std::move(edges_));
  }

private:
  GraphContextDescriptor ctx_desc_;
  std::unordered_map<std::string, IONode> io_nodes_;
  std::unordered_map<std::string, OpNode> op_nodes_;
  Vec<Edge> edges_;
  size_t node_count_ = 0;

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