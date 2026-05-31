#pragma once

#include <algorithm>
#include <initializer_list>
#include <iosfwd>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>

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
  int data_ref_count() const { return data_ref_count_; }
  void set_data(const Tensor &data, int ref_count) {
    data_ = data;
    data_ref_count_ = ref_count;
  }
  bool decrement_data_ref_count() {
    if (data_ref_count_ > 0) {
      --data_ref_count_;
      if (data_ref_count_ == 0) {
        data_ = Tensor();  // Release the tensor when ref count reaches zero
        return true;
      }
    }
    return false;
  }

  const Tensor &grad() const { return grad_; }
  int grad_ref_count() const { return grad_ref_count_; }
  void set_grad(const Tensor &grad, int ref_count) {
    grad_ = grad;
    grad_ref_count_ = ref_count;
  }
  bool decrement_grad_ref_count() {
    if (grad_ref_count_ > 0) {
      --grad_ref_count_;
      if (grad_ref_count_ == 0) {
        grad_ = Tensor();  // Release the tensor when ref count reaches zero
        return true;
      }
    }
    return false;
  }

private:
  Graph *graph_;
  std::string uid_;
  Tensor data_;
  int data_ref_count_ = 0;
  Tensor grad_;
  int grad_ref_count_ = 0;
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

  void compile(IAllocator &allocator) {
    sort();
    forward_plan_cache_.clear();
    GraphContextDescriptor ctx_desc;
    std::set<LayerImpl *> unique_layers;
    for (const auto &edge : edges_) {
      LayerImpl *layer_ptr = edge->layer().get();
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
    workspace_allocator_ = DELAllocatorV2::create(context_->device(), defaultFlowHandle);
    for (LayerImpl *layer_ptr : unique_layers) {
      layer_ptr->set_engine_type(allocator.device().get_engine());
      layer_ptr->set_allocator(*workspace_allocator_);
      layer_ptr->init();
    }
  }

  Vec<Node> nodes() const { return nodes_; }
  Vec<Edge> edges() const { return edges_; }
  const Device &device() const { return context_->device(); }
  size_t cached_forward_plan_count() const { return forward_plan_cache_.size(); }
  bool has_cached_forward_plan(const InputMap &input_map) const {
    return find_forward_plan(input_map) != nullptr;
  }
  size_t cached_forward_plan_execution_count(const InputMap &input_map) const {
    const ForwardPlanCacheEntry *plan = find_forward_plan(input_map);
    return plan ? plan->execution_count : 0;
  }
  bool cached_forward_plan_profiled(const InputMap &input_map) const {
    const ForwardPlanCacheEntry *plan = find_forward_plan(input_map);
    return plan ? plan->profiled : false;
  }
  size_t cached_forward_plan_spsc_region_count(const InputMap &input_map) const {
    const ForwardPlanCacheEntry *plan = find_forward_plan(input_map);
    return plan ? plan->regions.spsc_regions.size() : 0;
  }
  size_t cached_forward_plan_mpsc_region_count(const InputMap &input_map) const {
    const ForwardPlanCacheEntry *plan = find_forward_plan(input_map);
    return plan ? plan->regions.mpsc_regions.size() : 0;
  }
  size_t cached_forward_plan_profiled_edge_count(const InputMap &input_map) const {
    const ForwardPlanCacheEntry *plan = find_forward_plan(input_map);
    return plan ? plan->edge_profiles.size() : 0;
  }
  const DELAllocatorV2 *workspace_allocator() const { return workspace_allocator_.get(); }
  GraphRegionSummary classify_regions() {
    sort();

    GraphRegionSummary summary;
    std::unordered_map<NodeImpl *, size_t> incoming_edge_index;
    std::unordered_map<NodeImpl *, Vec<size_t>> outgoing_edge_indices;

    for (size_t edge_index = 0; edge_index < edges_.size(); ++edge_index) {
      const Edge &edge = edges_[edge_index];
      for (const auto &producer : edge->producers()) {
        outgoing_edge_indices[producer.get()].push_back(edge_index);
      }
      for (const auto &consumer : edge->consumers()) {
        if (node_in_degree(consumer) == 1) {
          incoming_edge_index[consumer.get()] = edge_index;
        }
      }
    }

    std::vector<bool> claimed(edges_.size(), false);

    auto is_spsc_edge = [this](size_t edge_index) {
      const Edge &edge = edges_[edge_index];
      return edge->producers().size() == 1 && edge->consumers().size() == 1;
    };

    for (size_t edge_index = 0; edge_index < edges_.size(); ++edge_index) {
      const Edge &join_edge = edges_[edge_index];
      if (join_edge->producers().size() < 2 || join_edge->consumers().size() != 1) {
        continue;
      }

      MPSCRegion region;
      region.join_edge_index = edge_index;
      bool valid = true;

      for (const auto &producer : join_edge->producers()) {
        if (node_out_degree(producer) != 1) {
          valid = false;
          break;
        }

        Vec<size_t> branch;
        Node current = producer;
        while (true) {
          auto incoming_it = incoming_edge_index.find(current.get());
          if (incoming_it == incoming_edge_index.end()) {
            break;
          }

          size_t predecessor_index = incoming_it->second;
          if (claimed[predecessor_index] || !is_spsc_edge(predecessor_index)) {
            break;
          }

          const Edge &predecessor = edges_[predecessor_index];
          if (predecessor->consumers()[0].get() != current.get()) {
            break;
          }

          branch.push_back(predecessor_index);

          Node previous = predecessor->producers()[0];
          current = previous;
          if (node_in_degree(current) != 1 || node_out_degree(current) != 1) {
            break;
          }
        }

        std::reverse(branch.begin(), branch.end());
        if (branch.empty()) {
          valid = false;
          break;
        }

        region.branch_edge_indices.push_back(branch);
      }

      if (!valid) {
        continue;
      }

      claimed[edge_index] = true;
      for (const auto &branch : region.branch_edge_indices) {
        for (size_t branch_edge_index : branch) {
          claimed[branch_edge_index] = true;
        }
      }
      summary.mpsc_regions.push_back(std::move(region));
    }

    for (size_t edge_index = 0; edge_index < edges_.size(); ++edge_index) {
      if (claimed[edge_index] || !is_spsc_edge(edge_index)) {
        continue;
      }

      const Edge &edge = edges_[edge_index];
      Node start = edge->producers()[0];
      bool has_spsc_predecessor = false;
      auto incoming_it = incoming_edge_index.find(start.get());
      if (incoming_it != incoming_edge_index.end()) {
        size_t predecessor_index = incoming_it->second;
        if (!claimed[predecessor_index] && is_spsc_edge(predecessor_index) &&
            edges_[predecessor_index]->consumers()[0].get() == start.get() &&
            node_out_degree(start) == 1) {
          has_spsc_predecessor = true;
        }
      }

      if (has_spsc_predecessor) {
        continue;
      }

      SPSCRegion region;
      size_t current_index = edge_index;
      while (true) {
        if (claimed[current_index] || !is_spsc_edge(current_index)) {
          break;
        }

        region.edge_indices.push_back(current_index);
        claimed[current_index] = true;

        Node consumer = edges_[current_index]->consumers()[0];
        if (node_in_degree(consumer) != 1 || node_out_degree(consumer) != 1) {
          break;
        }

        auto outgoing_it = outgoing_edge_indices.find(consumer.get());
        if (outgoing_it == outgoing_edge_indices.end() || outgoing_it->second.size() != 1) {
          break;
        }

        size_t next_index = outgoing_it->second.front();
        if (claimed[next_index] || !is_spsc_edge(next_index) ||
            edges_[next_index]->producers()[0].get() != consumer.get()) {
          break;
        }

        current_index = next_index;
      }

      if (!region.edge_indices.empty()) {
        summary.spsc_regions.push_back(std::move(region));
      }
    }

    for (size_t edge_index = 0; edge_index < edges_.size(); ++edge_index) {
      if (!claimed[edge_index]) {
        summary.unsupported_edge_indices.push_back(edge_index);
      }
    }

    return summary;
  }

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
    ForwardPlanCacheEntry &plan = get_or_create_forward_plan(input_map);
    std::map<std::string, Node> uid_to_node;
    for (const auto &node : nodes_) {
      uid_to_node[node->uid()] = node;
    }
    for (const auto &[uid, tensor] : input_map) {
      auto it = uid_to_node.find(uid);
      if (it == uid_to_node.end()) {
        throw std::runtime_error("Input UID not found in graph: " + uid);
      }
      auto node = it->second;
      it->second->set_data(tensor, out_degree_[node]);
    }

    size_t hook_id = 0;
    bool hook_registered = false;
    size_t edge_peak_usage = workspace_allocator_ ? workspace_allocator_->total_allocated() : 0;
    if (workspace_allocator_) {
      hook_id = workspace_allocator_->add_allocation_hook([&edge_peak_usage](size_t current_usage) {
        edge_peak_usage = std::max(edge_peak_usage, current_usage);
      });
      hook_registered = true;
    }

    for (size_t edge_index = 0; edge_index < edges_.size(); ++edge_index) {
      size_t usage_before = workspace_allocator_ ? workspace_allocator_->total_allocated() : 0;
      edge_peak_usage = usage_before;
      forward_edge(edges_[edge_index], mb_id);
      size_t usage_after = workspace_allocator_ ? workspace_allocator_->total_allocated() : 0;

      EdgeExecutionProfile &profile = plan.edge_profiles[edge_index];
      profile.peak_bytes =
          std::max(profile.peak_bytes,
                   edge_peak_usage >= usage_before ? edge_peak_usage - usage_before : size_t{0});
      profile.retained_bytes =
          std::max(profile.retained_bytes,
                   usage_after >= usage_before ? usage_after - usage_before : size_t{0});
    }

    if (hook_registered) {
      workspace_allocator_->remove_allocation_hook(hook_id);
    }

    OutputMap output_map;
    auto outputs = this->outputs();
    for (const auto &node : outputs) {
      output_map.set(node->uid(), node->data());
    }
    plan.execution_count++;
    plan.profiled = true;
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
      auto node = it->second;
      it->second->set_grad(tensor, in_degree_[node]);
    }
    for (auto it = edges_.rbegin(); it != edges_.rend(); ++it) {
      backward(*it, mb_id);
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
    mode_ = mode;
    for (const auto &edge : edges_) {
      edge->layer()->set_training(mode == ExecutionMode::TRAIN);
    }
  }

  Node input(const std::string &uid = "input") { return make_node(uid); }

  template <typename... Args>
  std::array<Node, sizeof...(Args)> inputs(Args... uids) {
    return {make_node(uids)...};
  }

private:
  Vec<Node> nodes_;
  Vec<Edge> edges_;
  std::unique_ptr<GraphContext> context_;
  std::shared_ptr<DELAllocatorV2> workspace_allocator_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> in_degree_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> out_degree_;
  std::unordered_map<std::string, ForwardPlanCacheEntry> forward_plan_cache_;
  ExecutionMode mode_ = ExecutionMode::TRAIN;
  size_t node_count_ = 0;
  std::set<std::string> used_uids_;

  int node_in_degree(const Node &node) const {
    auto it = in_degree_.find(node);
    return it == in_degree_.end() ? 0 : it->second;
  }

  int node_out_degree(const Node &node) const {
    auto it = out_degree_.find(node);
    return it == out_degree_.end() ? 0 : it->second;
  }

  std::string make_forward_plan_key(const InputMap &input_map) const {
    std::ostringstream key_builder;
    key_builder << static_cast<int>(mode_);
    for (const auto &[uid, tensor] : input_map) {
      key_builder << "|" << uid << ":";
      if (!tensor) {
        key_builder << "null";
        continue;
      }
      key_builder << "[";
      const auto &shape = tensor->shape();
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
          key_builder << ",";
        }
        key_builder << shape[i];
      }
      key_builder << "]";
    }
    return key_builder.str();
  }

  const ForwardPlanCacheEntry *find_forward_plan(const InputMap &input_map) const {
    auto it = forward_plan_cache_.find(make_forward_plan_key(input_map));
    return it == forward_plan_cache_.end() ? nullptr : &it->second;
  }

  ForwardPlanCacheEntry &get_or_create_forward_plan(const InputMap &input_map) {
    std::string key = make_forward_plan_key(input_map);
    auto [it, inserted] = forward_plan_cache_.try_emplace(key);
    if (inserted) {
      it->second.mode = mode_;
      it->second.regions = classify_regions();
      for (const auto &[uid, tensor] : input_map) {
        if (tensor) {
          it->second.input_shapes[uid] = tensor->shape();
        } else {
          it->second.input_shapes[uid] = {};
        }
      }
    }
    return it->second;
  }

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

  void forward_edge(Edge &edge, size_t mb_id = 0) {
    Vec<ConstTensor> input_data;
    for (const auto &producer : edge->producers()) {
      // Accumulate data from producer to layer input
      if (!producer->data()) {
        throw std::runtime_error("Null input data while forwarding graph");
      }
      input_data.push_back(producer->data());
      producer->decrement_data_ref_count();
    }
    Vec<Tensor> output_data = edge->layer()->forward(input_data, mb_id);
    for (size_t i = 0; i < edge->consumers().size(); ++i) {
      Node consumer = edge->consumers()[i];
      consumer->set_data(output_data[i], out_degree_[consumer]);
    }
  }

  void backward(Edge &edge, size_t mb_id = 0) {
    Vec<ConstTensor> output_grads;
    for (const auto &consumer : edge->consumers()) {
      if (!consumer->grad()) {
        throw std::runtime_error("Null output gradient while backwarding graph");
      }
      output_grads.push_back(consumer->grad());
      consumer->decrement_grad_ref_count();
    }
    Vec<Tensor> input_grads = edge->layer()->backward(output_grads, mb_id);
    for (size_t i = 0; i < edge->producers().size(); ++i) {
      Node producer = edge->producers()[i];
      producer->set_grad(input_grads[i], in_degree_[producer]);
    }
  }
};

}  // namespace tnn