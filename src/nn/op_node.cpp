#include "nn/op_node.hpp"

#include "nn/layers.hpp"

namespace tnn {
NodeConfig OpNode::get_config() const {
  NodeConfig config;
  config.type = "op_node";
  config.set("uid", uid());
  if (layer_) {
    config.set("name", layer_->name());
    config.set("layer_config", layer_->get_config().to_json());
  }

  std::vector<std::string> inputs;
  for (const auto* node : inputs_) {
    inputs.push_back(node->uid());
  }
  config.set("inputs", inputs);

  std::vector<std::string> outputs;
  for (const auto* node : outputs_) {
    outputs.push_back(node->uid());
  }
  config.set("outputs", outputs);

  return config;
}
OpNode OpNode::create_from_config(GraphContextDescriptor& ctx_desc, const NodeConfig& config) {
  nlohmann::json layer_json = config.get<nlohmann::json>("layer_config");
  LayerConfig layer_cfg = LayerConfig::from_json(layer_json);
  auto layer = LayerFactory::create(layer_cfg);
  std::string uid = config.get<std::string>("uid");
  OpNode node(uid, ctx_desc, std::move(layer));
  return node;
}

}  // namespace tnn