#pragma once

#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/io_node.hpp"
#include "nn/layers.hpp"
#include "nn/op_node.hpp"

namespace tnn {

using GraphConfig = TConfig;

class Graph {
public:
  Graph(IAllocator& allocator, GraphContextDescriptor ctx_desc,
        std::unordered_map<std::string, OpNode>&& op_nodes,
        std::unordered_map<std::string, IONode>&& io_nodes,
        std::vector<OpNode*>&& execution_sequence)
      : ctx_(allocator, ctx_desc),
        op_nodes_(std::move(op_nodes)),
        io_nodes_(std::move(io_nodes)),
        execution_sequence_(std::move(execution_sequence)) {
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
  const std::vector<OpNode*>& ops() const { return execution_sequence_; }

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
    std::vector<std::string> execution_sequence;
    for (const auto* node : execution_sequence_) {
      execution_sequence.push_back(node->uid());
    }
    config.set("execution_sequence", execution_sequence);
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
      OpNode node = OpNode::create_from_config(ctx_desc, op_cfg);
      op_nodes.emplace(node.uid(), std::move(node));
    }

    // Wire edges between OpNodes and IONodes
    for (const auto& op_j : op_json) {
      NodeConfig op_cfg = NodeConfig::from_json(op_j);
      OpNode* op_ptr = &op_nodes.at(op_cfg.get<std::string>("uid"));

      for (const auto& uid : op_cfg.get<std::vector<std::string>>("inputs", {})) {
        IONode* io_ptr = &io_nodes.at(uid);
        op_ptr->add_input(io_ptr);
        io_ptr->add_consumer(op_ptr);
      }

      for (const auto& uid : op_cfg.get<std::vector<std::string>>("outputs", {})) {
        IONode* io_ptr = &io_nodes.at(uid);
        op_ptr->add_output(io_ptr);
        io_ptr->add_producer(op_ptr);
      }
    }

    // Reconstruct execution sequence
    auto exec_seq_uids = config.get<std::vector<std::string>>("execution_sequence", {});
    std::vector<OpNode*> execution_sequence;
    execution_sequence.reserve(exec_seq_uids.size());
    for (const auto& uid : exec_seq_uids) {
      execution_sequence.push_back(&op_nodes.at(uid));
    }

    return Graph(allocator, ctx_desc, std::move(op_nodes), std::move(io_nodes),
                 std::move(execution_sequence));
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
  std::vector<OpNode*> execution_sequence_;
};

}  // namespace tnn