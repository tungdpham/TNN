#pragma once

#include <algorithm>
#include <initializer_list>
#include <iosfwd>
#include <iostream>
#include <map>
#include <queue>
#include <set>

#include "nn/graph_context.hpp"
#include "nn/layer.hpp"
#include "type/type.hpp"

namespace tnn {
class NodeImpl;
class EdgeImpl;
class Graph;

using Node = std::shared_ptr<NodeImpl>;
using Edge = std::shared_ptr<EdgeImpl>;

class NodeImpl {
public:
  explicit NodeImpl(Graph *graph, const std::string &uid = "")
      : graph_(graph),
        uid_(uid) {}
  ~NodeImpl() = default;

  Graph *graph() const { return graph_; }

  const std::string &uid() const { return uid_; }
  void set_uid(const std::string &uid) { uid_ = uid; }

  const Tensor &data() const { return data_; }
  void set_data(const Tensor &data) { data_ = data; }

  const Tensor &grad() const { return grad_; }
  void set_grad(const Tensor &grad) { grad_ = grad; }

private:
  Graph *graph_;
  std::string uid_;
  Tensor data_;
  Tensor grad_;
};

class InputMap {
private:
  std::map<std::string, Tensor> inputs_;  // Map from uid -> Tensor

public:
  InputMap() = default;

  InputMap(std::initializer_list<std::pair<const std::string, Tensor>> init)
      : inputs_(init) {}

  InputMap(std::initializer_list<std::pair<Node, Tensor>> init) {
    for (const auto &pair : init) {
      inputs_[pair.first->uid()] = pair.second;
    }
  }

  auto begin() noexcept { return inputs_.begin(); }
  auto end() noexcept { return inputs_.end(); }

  auto begin() const noexcept { return inputs_.begin(); }
  auto end() const noexcept { return inputs_.end(); }
  auto cbegin() const noexcept { return inputs_.cbegin(); }
  auto cend() const noexcept { return inputs_.cend(); }

  Tensor &operator[](const std::string &name) { return inputs_[name]; }
  const Tensor &operator[](const std::string &name) const { return inputs_.at(name); }

  void set(const std::string &name, const Tensor &tensor) { inputs_[name] = tensor; }
  const Tensor &get(const std::string &name) const { return inputs_.at(name); }
  bool contains(const std::string &name) const { return inputs_.count(name) > 0; }
  size_t size() const { return inputs_.size(); }
  void clear() { inputs_.clear(); }
};

using OutputMap = InputMap;

class EdgeImpl {
public:
  EdgeImpl(std::shared_ptr<LayerImpl> layer, const Vec<Node> &inputs, const Vec<Node> &outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  EdgeImpl(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> inputs,
           std::initializer_list<Node> outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  void forward(size_t mb_id = 0) {
    Vec<ConstTensor> input_data;
    for (const auto &producer : producers_) {
      // Accumulate data from producer to layer input
      if (!producer->data()) {
        throw std::runtime_error("Null input data while forwarding graph");
      }
      input_data.push_back(producer->data());
    }
    Vec<Tensor> output_data = layer_->forward(input_data, mb_id);
    for (size_t i = 0; i < consumers_.size(); ++i) {
      consumers_[i]->set_data(output_data[i]);
    }
  }

  void backward(size_t mb_id = 0) {
    Vec<ConstTensor> output_grads;
    for (const auto &consumer : consumers_) {
      if (!consumer->grad()) {
        throw std::runtime_error("Null output gradient while backwarding graph");
      }
      output_grads.push_back(consumer->grad());
    }
    Vec<Tensor> input_grads = layer_->backward(output_grads, mb_id);
    for (size_t i = 0; i < producers_.size(); ++i) {
      producers_[i]->set_grad(input_grads[i]);
    }
  }

  const Vec<Node> &producers() const { return producers_; }
  const Vec<Node> &consumers() const { return consumers_; }
  std::shared_ptr<LayerImpl> layer() const { return layer_; }

private:
  std::shared_ptr<LayerImpl> layer_;
  Vec<Node> producers_;
  Vec<Node> consumers_;
};

enum class ExecutionMode {
  TRAIN,
  EVAL,
};

class Graph {
public:
  Graph() = default;

  void save_state(std::ostream &stream) const;
  static Graph load_state(std::istream &stream, IAllocator &allocator);

  void compile(IAllocator &allocator) {
    sort();
    GraphContextDescriptor ctx_desc;
    std::set<LayerImpl *> unique_layers;
    for (const auto &edge : edges_) {
      LayerImpl *layer_ptr = edge->layer().get();
      std::cout << fmt::format("Registering layer in graph context: {}", layer_ptr->name())
                << std::endl;
      if (unique_layers.count(layer_ptr) == 0) {
        unique_layers.insert(layer_ptr);
      }
    }
    for (LayerImpl *layer_ptr : unique_layers) {
      for (const auto &param_desc : layer_ptr->param_descriptors()) {
        ctx_desc.register_desc(param_desc);
      }
    }
    context_ = std::make_unique<GraphContext>(allocator, ctx_desc);
    workspace_allocator_ = DELAllocatorV2::instance(context_->device(), defaultFlowHandle);
    for (LayerImpl *layer_ptr : unique_layers) {
      std::cout << fmt::format("Initializing layer: {}", layer_ptr->name()) << std::endl;
      layer_ptr->set_engine_type(allocator.device().get_engine());
      layer_ptr->set_allocator(*workspace_allocator_);
      layer_ptr->init();
    }
    build_execution_plan();
  }

  Vec<Node> nodes() const { return nodes_; }
  Vec<Edge> edges() const { return edges_; }
  const Device &device() const { return context_->device(); }

  void add_edge(std::shared_ptr<LayerImpl> layer, const Vec<Node> &producers,
                const Vec<Node> &consumers) {
    Edge edge = std::make_shared<EdgeImpl>(layer, producers, consumers);
    edges_.push_back(edge);
    on_add_edge(edge);
  }

  void add_edge(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> producers,
                std::initializer_list<Node> consumers) {
    Edge edge = std::make_shared<EdgeImpl>(layer, producers, consumers);
    edges_.push_back(edge);
    on_add_edge(edge);
  }

  void sort() {
    std::set<std::weak_ptr<NodeImpl>, std::owner_less<std::weak_ptr<NodeImpl>>> produced_nodes;
    for (const auto &edge : edges_) {
      for (const auto &consumer : edge->consumers()) {
        produced_nodes.insert(consumer);
      }
    }

    std::map<std::weak_ptr<NodeImpl>, Vec<Edge>, std::owner_less<std::weak_ptr<NodeImpl>>>
        node_to_dependent_edges;
    for (const auto &edge : edges_) {
      for (const auto &producer : edge->producers()) {
        node_to_dependent_edges[producer].push_back(edge);
      }
    }

    std::map<Edge, int> pending;
    std::queue<Edge> ready;

    for (const auto &edge : edges_) {
      int count = 0;
      for (const auto &producer : edge->producers()) {
        if (produced_nodes.count(producer)) {
          ++count;
        }
      }
      pending[edge] = count;
      if (count == 0) {
        ready.push(edge);
      }
    }

    Vec<Edge> sorted;
    sorted.reserve(edges_.size());

    while (!ready.empty()) {
      Edge e = ready.front();
      ready.pop();
      sorted.push_back(e);

      for (const auto &consumer : e->consumers()) {
        auto it = node_to_dependent_edges.find(consumer);
        if (it != node_to_dependent_edges.end()) {
          for (const auto &dep_edge : it->second) {
            if (--pending[dep_edge] == 0) {
              ready.push(dep_edge);
            }
          }
        }
      }
    }

    if (sorted.size() != edges_.size()) {
      throw std::runtime_error("Graph contains a cycle; topological sort failed");
    }

    edges_ = std::move(sorted);

    std::set<NodeImpl *> placed;
    Vec<Node> sorted_nodes;
    sorted_nodes.reserve(nodes_.size());

    for (const auto &node : nodes_) {
      if (!produced_nodes.count(node)) {
        sorted_nodes.push_back(node);
        placed.insert(node.get());
      }
    }

    for (const auto &edge : edges_) {
      for (const auto &consumer : edge->consumers()) {
        if (placed.insert(consumer.get()).second) {
          sorted_nodes.push_back(consumer);
        }
      }
    }

    nodes_ = std::move(sorted_nodes);
  }

  OutputMap forward(InputMap &input_map, size_t mb_id = 0) {
    std::map<std::string, Node> uid_to_node;
    for (const auto &node : nodes_) {
      uid_to_node[node->uid()] = node;
    }
    for (const auto &[uid, tensor] : input_map) {
      auto it = uid_to_node.find(uid);
      if (it == uid_to_node.end()) {
        throw std::runtime_error("Input UID not found in graph: " + uid);
      }
      it->second->set_data(tensor);
    }
    switch (execution_plan_kind_) {
      case ExecutionPlanKind::Spsc:
        execute_spsc_chain_forward(spsc_chain_, mb_id,
                                   workspace_allocator_ ? workspace_allocator_->side() : 0);
        break;
      case ExecutionPlanKind::SingleJoin:
        execute_single_join_forward(mb_id);
        break;
      case ExecutionPlanKind::Topological:
      default:
        for (auto it = edges_.begin(); it != edges_.end(); ++it) {
          (*it)->forward(mb_id);
        }
        break;
    }
    OutputMap output_map;
    auto outputs = this->outputs();
    for (const auto &node : outputs) {
      output_map.set(node->uid(), node->data());
    }
    return output_map;
  }

  OutputMap backward(InputMap &output_grad_map, size_t mb_id = 0) {
    std::map<std::string, Node> uid_to_node;
    for (const auto &node : nodes_) {
      uid_to_node[node->uid()] = node;
    }
    for (const auto &[uid, tensor] : output_grad_map) {
      auto it = uid_to_node.find(uid);
      if (it == uid_to_node.end()) {
        throw std::runtime_error("Output UID not found in graph: " + uid);
      }
      it->second->set_grad(tensor);
    }
    switch (execution_plan_kind_) {
      case ExecutionPlanKind::Spsc:
        execute_spsc_chain_backward(spsc_chain_, mb_id,
                                    workspace_allocator_ ? workspace_allocator_->side() : 0);
        break;
      case ExecutionPlanKind::SingleJoin:
        execute_single_join_backward(mb_id);
        break;
      case ExecutionPlanKind::Topological:
      default:
        for (auto it = edges_.rbegin(); it != edges_.rend(); ++it) {
          (*it)->backward(mb_id);
        }
        break;
    }
    OutputMap input_grad_map;
    auto inputs = this->inputs();
    for (const auto &node : inputs) {
      input_grad_map.set(node->uid(), node->grad());
    }
    return input_grad_map;
  }

  Node make_node(std::string uid = "") {
    if (uid.empty()) {
      uid = generate_uid();
    } else if (used_uids_.count(uid) > 0) {
      throw std::runtime_error("Duplicate node UID: " + uid);
    } else {
      used_uids_.insert(uid);
    }
    Node node = std::make_shared<NodeImpl>(this, uid);
    nodes_.push_back(node);
    return node;
  }

  GraphContext *context() const { return context_.get(); }

  void set_mode(ExecutionMode mode) {
    for (const auto &edge : edges_) {
      edge->layer()->set_training(mode == ExecutionMode::TRAIN);
    }
  }

private:
  enum class ExecutionPlanKind {
    Topological,
    Spsc,
    SingleJoin,
  };

  struct SingleJoinPlan {
    Vec<Vec<Edge>> branches;
    Edge join_edge;
    Vec<Edge> tail;
  };

  Vec<Node> nodes_;
  Vec<Edge> edges_;
  std::unique_ptr<GraphContext> context_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> in_degree_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> out_degree_;
  size_t node_count_ = 0;
  std::set<std::string> used_uids_;
  std::shared_ptr<DELAllocatorV2> workspace_allocator_;
  ExecutionPlanKind execution_plan_kind_ = ExecutionPlanKind::Topological;
  Vec<Edge> spsc_chain_;
  SingleJoinPlan single_join_plan_;
  Vec<size_t> single_join_execution_order_;
  bool single_join_execution_order_cached_ = false;

  std::string generate_uid() {
    std::string uid;
    do {
      uid = "node_" + std::to_string(node_count_++);
    } while (used_uids_.count(uid) > 0);
    used_uids_.insert(uid);
    return uid;
  }

  Vec<Node> inputs() {
    Vec<Node> input_nodes;
    for (const auto &node : nodes_) {
      auto it = in_degree_.find(node);
      // If it's not in the map, its in-degree is effectively 0
      if (it == in_degree_.end() || it->second == 0) {
        input_nodes.push_back(node);
      }
    }
    return input_nodes;
  }

  Vec<Node> outputs() {
    Vec<Node> output_nodes;
    for (const auto &node : nodes_) {
      auto it = out_degree_.find(node);
      // If it's not in the map, its out-degree is effectively 0
      if (it == out_degree_.end() || it->second == 0) {
        output_nodes.push_back(node);
      }
    }
    return output_nodes;
  }

  void on_add_edge(const Edge &edge) {
    for (const auto &producer : edge->producers()) {
      out_degree_[producer]++;
    }
    for (const auto &consumer : edge->consumers()) {
      in_degree_[consumer]++;
    }
  }

  void build_execution_plan() {
    execution_plan_kind_ = ExecutionPlanKind::Topological;
    spsc_chain_.clear();
    single_join_plan_ = SingleJoinPlan{};
    single_join_execution_order_.clear();
    single_join_execution_order_cached_ = false;

    if (try_build_spsc_plan()) {
      execution_plan_kind_ = ExecutionPlanKind::Spsc;
      return;
    }

    if (try_build_single_join_plan()) {
      execution_plan_kind_ = ExecutionPlanKind::SingleJoin;
    }
  }

  bool try_build_spsc_plan() {
    if (edges_.empty() || inputs().size() != 1 || outputs().size() != 1) {
      return false;
    }

    for (const auto &edge : edges_) {
      if (edge->producers().size() != 1 || edge->consumers().size() != 1) {
        return false;
      }
    }

    for (const auto &node : nodes_) {
      const bool is_input = !in_degree_.count(node) || in_degree_.at(node) == 0;
      const bool is_output = !out_degree_.count(node) || out_degree_.at(node) == 0;
      if (is_input || is_output) {
        continue;
      }
      if (in_degree_.at(node) != 1 || out_degree_.at(node) != 1) {
        return false;
      }
    }

    spsc_chain_ = edges_;
    return true;
  }

  bool try_build_single_join_plan() {
    if (edges_.empty() || outputs().size() != 1) {
      return false;
    }

    std::map<NodeImpl *, Edge> producer_edge_by_output;
    std::map<NodeImpl *, Vec<Edge>> consumer_edges_by_node;
    Vec<Edge> multi_input_edges;

    for (const auto &edge : edges_) {
      if (edge->consumers().size() != 1) {
        return false;
      }
      if (edge->producers().size() > 1) {
        multi_input_edges.push_back(edge);
      }
      for (const auto &consumer : edge->consumers()) {
        producer_edge_by_output[consumer.get()] = edge;
      }
      for (const auto &producer : edge->producers()) {
        consumer_edges_by_node[producer.get()].push_back(edge);
      }
    }

    if (multi_input_edges.size() != 1) {
      return false;
    }

    Edge join_edge = multi_input_edges.front();
    std::set<Edge> covered_edges;
    Vec<Vec<Edge>> branches;
    branches.reserve(join_edge->producers().size());

    for (const auto &join_input : join_edge->producers()) {
      Vec<Edge> branch;
      Node current = join_input;
      while (true) {
        auto producer_it = producer_edge_by_output.find(current.get());
        if (producer_it == producer_edge_by_output.end()) {
          break;
        }

        Edge edge = producer_it->second;
        if (edge == join_edge || edge->producers().size() != 1 || edge->consumers().size() != 1) {
          return false;
        }

        Node producer = edge->producers().front();
        if ((!out_degree_.count(producer) || out_degree_.at(producer) == 0) ||
            out_degree_.at(producer) > 1) {
          return false;
        }
        if (in_degree_.count(current) && in_degree_.at(current) > 1 && current != join_input) {
          return false;
        }

        branch.push_back(edge);
        covered_edges.insert(edge);
        current = producer;
      }
      std::reverse(branch.begin(), branch.end());
      branches.push_back(std::move(branch));
    }

    Vec<Edge> tail;
    Node current = join_edge->consumers().front();
    while (true) {
      auto consumer_it = consumer_edges_by_node.find(current.get());
      if (consumer_it == consumer_edges_by_node.end()) {
        break;
      }
      if (consumer_it->second.size() != 1 || out_degree_.at(current) > 1) {
        return false;
      }

      Edge edge = consumer_it->second.front();
      if (edge->producers().size() != 1 || edge->consumers().size() != 1) {
        return false;
      }
      if (in_degree_.count(current) && in_degree_.at(current) != 1) {
        return false;
      }

      tail.push_back(edge);
      covered_edges.insert(edge);
      current = edge->consumers().front();
    }

    covered_edges.insert(join_edge);
    if (covered_edges.size() != edges_.size()) {
      return false;
    }

    single_join_plan_.branches = std::move(branches);
    single_join_plan_.join_edge = join_edge;
    single_join_plan_.tail = std::move(tail);
    return true;
  }

  void execute_spsc_chain_forward(const Vec<Edge> &chain, size_t mb_id, int output_side) {
    if (chain.empty()) {
      return;
    }
    if (!workspace_allocator_) {
      for (const auto &edge : chain) {
        edge->forward(mb_id);
      }
      return;
    }

    workspace_allocator_->set_side(output_side);
    if (chain.size() % 2 == 0) {
      workspace_allocator_->flip();
    }
    for (size_t i = 0; i < chain.size(); ++i) {
      chain[i]->forward(mb_id);
      if (i + 1 < chain.size()) {
        workspace_allocator_->flip();
      }
    }
  }

  void execute_spsc_chain_backward(const Vec<Edge> &chain, size_t mb_id, int input_grad_side) {
    if (chain.empty()) {
      return;
    }
    if (!workspace_allocator_) {
      for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        (*it)->backward(mb_id);
      }
      return;
    }

    workspace_allocator_->set_side(input_grad_side);
    if (chain.size() % 2 == 0) {
      workspace_allocator_->flip();
    }
    for (size_t offset = 0; offset < chain.size(); ++offset) {
      size_t index = chain.size() - 1 - offset;
      chain[index]->backward(mb_id);
      if (offset + 1 < chain.size()) {
        workspace_allocator_->flip();
      }
    }
  }

  void clear_forward_measurement_artifacts(const Vec<Edge> &chain, size_t mb_id) {
    for (const auto &edge : chain) {
      edge->layer()->clear_cache(mb_id);
    }
    for (const auto &edge : chain) {
      for (const auto &consumer : edge->consumers()) {
        consumer->set_data(Tensor());
      }
    }
  }

  std::pair<size_t, size_t> measure_chain_memory(const Vec<Edge> &chain, size_t mb_id,
                                                 int output_side) {
    if (chain.empty() || !workspace_allocator_) {
      return {0, 0};
    }

    int original_side = workspace_allocator_->side();
    size_t m_prev = workspace_allocator_->total_allocated();
    size_t m_max = m_prev;
    size_t hook_id = workspace_allocator_->add_allocation_hook([&m_max](size_t total_allocated) {
      if (total_allocated > m_max) {
        m_max = total_allocated;
      }
    });

    execute_spsc_chain_forward(chain, mb_id, output_side);
    context_->device().getFlow(defaultFlowHandle)->synchronize();
    workspace_allocator_->remove_allocation_hook(hook_id);

    size_t m_after = workspace_allocator_->total_allocated();
    clear_forward_measurement_artifacts(chain, mb_id);
    workspace_allocator_->set_side(original_side);

    return {m_max - m_prev, m_after - m_prev};
  }

  Vec<size_t> compute_single_join_execution_order(size_t mb_id) {
    if (single_join_execution_order_cached_) {
      return single_join_execution_order_;
    }

    struct BranchMemInfo {
      size_t index;
      size_t cycling_cost;
      size_t retained_cost;
      long long priority;
    };

    Vec<BranchMemInfo> infos;
    infos.reserve(single_join_plan_.branches.size());
    int branch_output_side = workspace_allocator_ ? workspace_allocator_->side() : 0;

    for (size_t i = 0; i < single_join_plan_.branches.size(); ++i) {
      const auto &branch = single_join_plan_.branches[i];
      if (branch.empty()) {
        infos.push_back({i, 0, 0, 0});
        continue;
      }
      auto [cycling_cost, retained_cost] = measure_chain_memory(branch, mb_id, branch_output_side);
      infos.push_back(
          {i, cycling_cost, retained_cost,
           static_cast<long long>(cycling_cost) - static_cast<long long>(retained_cost)});
    }

    std::sort(infos.begin(), infos.end(), [](const BranchMemInfo &a, const BranchMemInfo &b) {
      return a.priority > b.priority;
    });

    single_join_execution_order_.clear();
    for (const auto &info : infos) {
      single_join_execution_order_.push_back(info.index);
    }
    single_join_execution_order_cached_ = true;
    return single_join_execution_order_;
  }

  void execute_single_join_forward(size_t mb_id) {
    int branch_output_side = workspace_allocator_ ? workspace_allocator_->side() : 0;
    Vec<size_t> order = compute_single_join_execution_order(mb_id);

    for (size_t branch_index : order) {
      execute_spsc_chain_forward(single_join_plan_.branches[branch_index], mb_id,
                                 branch_output_side);
    }

    if (workspace_allocator_) {
      workspace_allocator_->set_side(1 - branch_output_side);
    }
    single_join_plan_.join_edge->forward(mb_id);
    execute_spsc_chain_forward(single_join_plan_.tail, mb_id, branch_output_side);
  }

  void execute_single_join_backward(size_t mb_id) {
    int join_input_grad_side = workspace_allocator_ ? workspace_allocator_->side() : 0;
    execute_spsc_chain_backward(single_join_plan_.tail, mb_id, join_input_grad_side);

    if (workspace_allocator_) {
      workspace_allocator_->set_side(1 - join_input_grad_side);
    }
    single_join_plan_.join_edge->backward(mb_id);

    Vec<size_t> order = compute_single_join_execution_order(mb_id);
    for (size_t offset = 0; offset < order.size(); ++offset) {
      size_t branch_index = order[order.size() - 1 - offset];
      execute_spsc_chain_backward(single_join_plan_.branches[branch_index], mb_id,
                                  join_input_grad_side);
    }
  }
};

}  // namespace tnn