#pragma once

#include <initializer_list>
#include <map>
#include <queue>

#include "nn/graph_context.hpp"
#include "nn/layer.hpp"
#include "type/type.hpp"

namespace tnn {
namespace graph_api_v2 {
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

  void set(const std::string &name, const Tensor &tensor) { inputs_[name] = tensor; }
  const Tensor &get(const std::string &name) const { return inputs_.at(name); }
  bool contains(const std::string &name) const { return inputs_.count(name) > 0; }
  size_t size() const { return inputs_.size(); }
  void clear() { inputs_.clear(); }
};

using OutputMap = InputMap;

class EdgeImpl {
public:
  EdgeImpl(std::shared_ptr<Layer> layer, const Vec<Node> &inputs, const Vec<Node> &outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  EdgeImpl(std::shared_ptr<Layer> layer, std::initializer_list<Node> inputs,
           std::initializer_list<Node> outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  void forward() {
    Vec<ConstTensor> input_data;
    for (const auto &producer : producers_) {
      // Accumulate data from producer to layer input
      if (!producer->data()) {
        throw std::runtime_error("Null input data while forwarding graph");
      }
      input_data.push_back(producer->data());
    }

    Vec<Tensor> output_data = layer_->forward(input_data);

    for (size_t i = 0; i < consumers_.size(); ++i) {
      consumers_[i]->set_data(output_data[i]);
    }
  }

  void backward() {
    Vec<ConstTensor> output_grads;
    for (const auto &consumer : consumers_) {
      if (!consumer->grad()) {
        throw std::runtime_error("Null output gradient while backwarding graph");
      }
      output_grads.push_back(consumer->grad());
    }

    Vec<Tensor> input_grads = layer_->backward(output_grads);

    for (size_t i = 0; i < producers_.size(); ++i) {
      producers_[i]->set_grad(input_grads[i]);
    }
  }

  const Vec<Node> &producers() const { return producers_; }
  const Vec<Node> &consumers() const { return consumers_; }
  std::shared_ptr<Layer> layer() const { return layer_; }

private:
  std::shared_ptr<Layer> layer_;
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

  void compile(IAllocator &allocator) {
    sort();
    GraphContextDescriptor ctx_desc;
    std::set<Layer *> unique_layers;
    for (const auto &edge : edges_) {
      Layer *layer_ptr = edge->layer().get();
      if (unique_layers.count(layer_ptr) == 0) {
        unique_layers.insert(layer_ptr);
      }
    }
    for (Layer *layer_ptr : unique_layers) {
      for (const auto &param_desc : layer_ptr->param_descriptors()) {
        ctx_desc.register_desc(param_desc);
      }
    }
    context_ = std::make_unique<GraphContext>(allocator, ctx_desc);
    auto ws_allocator = DELAllocatorV2::instance(context_->device(), defaultFlowHandle);
    for (Layer *layer_ptr : unique_layers) {
      layer_ptr->set_engine_type(allocator.device().get_engine());
      layer_ptr->set_allocator(*ws_allocator);
      layer_ptr->init();
    }
  }

  Vec<Node> nodes() const { return nodes_; }
  Vec<Edge> edges() const { return edges_; }

  void add_edge(std::shared_ptr<Layer> layer, const Vec<Node> &producers,
                const Vec<Node> &consumers) {
    Edge edge = std::make_shared<EdgeImpl>(layer, producers, consumers);
    edges_.push_back(edge);
    on_add_edge(edge);
  }

  void add_edge(std::shared_ptr<Layer> layer, std::initializer_list<Node> producers,
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

  OutputMap forward(InputMap &input_map) {
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
    for (auto it = edges_.begin(); it != edges_.end(); ++it) {
      (*it)->forward();
    }
    OutputMap output_map;
    auto outputs = this->outputs();
    for (const auto &node : outputs) {
      output_map.set(node->uid(), node->data());
    }
    return output_map;
  }

  OutputMap backward(InputMap &output_grad_map) {
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
    for (auto it = edges_.rbegin(); it != edges_.rend(); ++it) {
      (*it)->backward();
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
  Vec<Node> nodes_;
  Vec<Edge> edges_;
  std::unique_ptr<GraphContext> context_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> in_degree_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> out_degree_;
  size_t node_count_ = 0;
  std::set<std::string> used_uids_;

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
};

template <typename LayerType>
class LayerRef {
public:
  LayerRef(std::shared_ptr<LayerType> layer)
      : layer_(layer) {}

  LayerType *operator->() const { return layer_.get(); }
  LayerType &operator*() const { return *layer_; }

  operator std::shared_ptr<LayerType>() const { return layer_; }
  std::shared_ptr<LayerType> get() const { return layer_; }

  template <typename... Args>
  decltype(auto) operator()(Args &&...args) const {
    if (!layer_) {
      throw std::runtime_error("LayerRef: underlying shared_ptr is null");
    }
    return (*layer_)(std::forward<Args>(args)...);
  }

private:
  std::shared_ptr<LayerType> layer_;
};

template <typename LayerType, typename... Args>
auto make_layer(Args &&...args) -> LayerRef<LayerType> {
  return LayerRef<LayerType>(std::make_shared<LayerType>(std::forward<Args>(args)...));
}

}  // namespace graph_api_v2
}  // namespace tnn