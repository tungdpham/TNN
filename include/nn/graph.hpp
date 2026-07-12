#pragma once

#include <array>
#include <initializer_list>
#include <iosfwd>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "nn/edge.hpp"
#include "nn/graph_context.hpp"
#include "nn/node.hpp"
#include "nn/tensor_bundle.hpp"
#include "type/type.hpp"

namespace tunx {

enum class ExecutionMode {
  TRAIN,
  EVAL,
};

struct SPSCRegion {
  Vec<size_t> edge_indices;
};

struct MPSCRegion {
  size_t join_edge_index = 0;
  Vec<Vec<size_t>> branch_edge_indices;
};

struct GraphRegionSummary {
  Vec<SPSCRegion> spsc_regions;
  Vec<MPSCRegion> mpsc_regions;
  Vec<size_t> unsupported_edge_indices;
};

struct EdgeExecutionProfile {
  size_t peak_bytes = 0;
  size_t retained_bytes = 0;
};

struct ForwardPlanCacheEntry {
  ExecutionMode mode = ExecutionMode::TRAIN;
  std::map<std::string, Vec<size_t>> input_shapes;
  GraphRegionSummary regions;
  std::map<size_t, EdgeExecutionProfile> edge_profiles;
  size_t execution_count = 0;
  bool profiled = false;
};

class Graph {
public:
  Graph() = default;

  void save_state(std::ostream &stream) const;
  static Graph load_state(std::istream &stream, IAllocator &allocator);

  void compile(IAllocator &allocator);

  Vec<Node> nodes() const { return nodes_; }
  Vec<Edge> edges() const { return edges_; }
  Vec<std::string> input_uids() const;
  Vec<std::string> output_uids() const;
  const Device &device() const { return context_->device(); }
  const DELAllocatorV2 *workspace_allocator() const { return workspace_allocator_.get(); }

  void add_edge(std::shared_ptr<LayerImpl> layer, const Vec<Node> &producers,
                const Vec<Node> &consumers);

  void add_edge(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> producers,
                std::initializer_list<Node> consumers);

  void sort();

  TensorBundle forward(TensorBundle &input_map, size_t pid = 0);
  TensorBundle backward(TensorBundle &output_grad_map, size_t pid = 0);

  Node make_node(std::string uid = "");

  GraphContext *context() const { return context_.get(); }

  void set_mode(ExecutionMode mode);
  void set_io_dtype(DType_t dtype);
  void set_param_dtype(DType_t dtype);
  void set_compute_dtype(DType_t dtype);

  void set_input(Node node);
  void set_output(Node node);

  bool is_input(const Node &node) const { return input_nodes_.count(node) > 0; }
  bool is_output(const Node &node) const { return output_nodes_.count(node) > 0; }

  Node input(const std::string &uid = "input");

  template <typename... Args>
  std::array<Node, sizeof...(Args)> inputs(Args... uids) {
    return {make_node(uids)...};
  }

  void zero_grads();

  Vec<Tensor *> parameters();
  Vec<Tensor *> gradients();

private:
  Vec<Node> nodes_;
  Vec<Edge> edges_;
  std::set<Node> input_nodes_;
  std::set<Node> output_nodes_;
  std::unique_ptr<GraphContext> context_;
  std::shared_ptr<DELAllocatorV2> workspace_allocator_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> in_degree_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> out_degree_;
  ExecutionMode mode_ = ExecutionMode::TRAIN;
  size_t node_count_ = 0;
  std::set<std::string> used_uids_;

  int node_in_degree(const Node &node) const;
  int node_out_degree(const Node &node) const;
  std::string generate_uid();

  Vec<Node> inputs();
  Vec<Node> outputs();

  void on_add_edge(const Edge &edge);
  void forward_edge(Edge &edge, size_t pid = 0);
  void backward_edge(Edge &edge, size_t pid = 0);
};

}  // namespace tunx