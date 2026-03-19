#pragma once

#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/edge.hpp"
#include "nn/io_node.hpp"
#include "nn/layers.hpp"
#include "nn/op_node.hpp"

namespace tnn {

using GraphConfig = TConfig;

class Graph {
public:
  Graph(IAllocator& allocator, GraphContextDescriptor ctx_desc,
        std::unordered_map<std::string, OpNode>&& op_nodes,
        std::unordered_map<std::string, IONode>&& io_nodes, std::vector<Edge>&& edges)
      : ctx_(allocator, ctx_desc),
        op_nodes_(std::move(op_nodes)),
        io_nodes_(std::move(io_nodes)),
        edges_(std::move(edges)) {
    for (auto& op_pair : op_nodes_) {
      OpNode& op_node = op_pair.second;
      op_node.init(allocator);
    }
  }

  Graph(const Graph& other) = delete;
  Graph(Graph&& other) = default;
  Graph& operator=(const Graph& other) = delete;
  Graph& operator=(Graph&& other) = delete;

  GraphContext& context() { return ctx_; }

  const Device& device() const { return ctx_.device(); }

  const std::unordered_map<std::string, IONode>& io_nodes() const { return io_nodes_; }
  const IONode& io_node(const std::string& uid) const { return io_nodes_.at(uid); }
  const std::vector<Edge>& edges() const { return edges_; }

  const std::string& name() const { return name_; }
  Graph& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }

  Graph& set_training(bool training) {
    for (auto& [uid, op_node] : op_nodes_) {
      op_node.set_training(training);
    }
    return *this;
  }

  // temporary: get layers from op nodes
  // need to refactor later if we want to support more complex graph structures
  std::vector<Layer*> get_layers() {
    std::vector<Layer*> layers;
    for (auto& op_pair : op_nodes_) {
      OpNode& node = op_pair.second;
      Sequential* seq = dynamic_cast<Sequential*>(node.layer());
      if (seq) {
        auto seq_layers = seq->get_layers();
        layers.insert(layers.end(), seq_layers.begin(), seq_layers.end());
        continue;
      } else {
        Layer* layer = dynamic_cast<Layer*>(node.layer());
        if (layer) layers.push_back(layer);
      }
    }
    return layers;
  }

  GraphConfig get_config() const {
    GraphConfig config;
    config.type = "graph";
    std::vector<nlohmann::json> op_configs;
    for (const auto& op_pair : op_nodes_) {
      const OpNode& node = op_pair.second;
      op_configs.push_back(node.get_config().to_json());
    }
    config.set("op_nodes", nlohmann::json(op_configs));
    std::vector<nlohmann::json> io_configs;
    for (const auto& io_pair : io_nodes_) {
      const IONode& node = io_pair.second;
      io_configs.push_back(node.get_config().to_json());
    }
    config.set("io_nodes", nlohmann::json(io_configs));
    std::vector<nlohmann::json> edge_configs;
    for (const auto& edge : edges_) {
      nlohmann::json edge_j;
      std::vector<std::string> producer_uids;
      for (const auto& producer : edge.producers()) {
        producer_uids.push_back(producer->uid());
      }
      edge_j["producers"] = producer_uids;
      std::vector<std::string> consumer_uids;
      for (const auto& consumer : edge.consumers()) {
        consumer_uids.push_back(consumer->uid());
      }
      edge_j["consumers"] = consumer_uids;
      // Use the op_node UID (unique) rather than the layer name (not unique)
      for (const auto& op_pair : op_nodes_) {
        if (&op_pair.second == &edge.op_node()) {
          edge_j["op_node"] = op_pair.first;
          break;
        }
      }
      edge_configs.push_back(edge_j);
    }
    config.set("edges", nlohmann::json(edge_configs));
    return config;
  }

  void save_state(std::ofstream& os) const {
    nlohmann::json json_config = get_config().to_json();
    os << json_config.dump(4);
    auto params = ctx_.parameters();
    for (const auto& param : params) {
      param->save(os);
    }
  }

  static Graph create_from_config(IAllocator& allocator, const GraphConfig& config) {
    LayerFactory::register_defaults();

    // Reconstruct IONodes
    std::unordered_map<std::string, IONode> io_nodes;
    nlohmann::json io_json = config.get<nlohmann::json>("io_nodes", nlohmann::json::array());
    for (const auto& io_j : io_json) {
      NodeConfig io_cfg = NodeConfig::from_json(io_j);
      IONode node = IONode::create_from_config(io_cfg);
      io_nodes.emplace(node.uid(), std::move(node));
    }

    // Reconstruct OpNodes (layers only; edges wired below)
    GraphContextDescriptor ctx_desc;
    std::unordered_map<std::string, OpNode> op_nodes;
    nlohmann::json op_json = config.get<nlohmann::json>("op_nodes", nlohmann::json::array());
    for (const auto& op_j : op_json) {
      NodeConfig op_cfg = NodeConfig::from_json(op_j);
      OpNode node = OpNode::create_from_config(op_cfg);
      op_nodes.emplace(node.uid(), std::move(node));
    }

    // Reconstruct edges
    std::vector<Edge> edges;
    nlohmann::json edge_json = config.get<nlohmann::json>("edges", nlohmann::json::array());
    for (const auto& edge_j : edge_json) {
      std::vector<const IONode*> producers;
      for (const auto& uid : edge_j.value("producers", std::vector<std::string>{})) {
        producers.push_back(&io_nodes.at(uid));
      }
      std::vector<const IONode*> consumers;
      for (const auto& uid : edge_j.value("consumers", std::vector<std::string>{})) {
        consumers.push_back(&io_nodes.at(uid));
      }
      std::string op_uid = edge_j.at("op_node").get<std::string>();
      edges.emplace_back(producers, consumers, op_nodes.at(op_uid));
    }

    return Graph(allocator, ctx_desc, std::move(op_nodes), std::move(io_nodes), std::move(edges));
  }

  static Graph load_state(std::ifstream& is, IAllocator& allocator) {
    nlohmann::json json_config;
    is >> json_config;
    GraphConfig config = GraphConfig::from_json(json_config);
    Graph graph = Graph::create_from_config(allocator, config);
    auto params = graph.ctx_.parameters();
    for (auto& param : params) {
      load_into(is, param);
    }
    graph.set_name(config.name);
    return graph;
  }

private:
  std::string name_;
  GraphContext ctx_;
  std::unordered_map<std::string, OpNode> op_nodes_;
  std::unordered_map<std::string, IONode> io_nodes_;
  std::vector<Edge> edges_;
};

}  // namespace tnn