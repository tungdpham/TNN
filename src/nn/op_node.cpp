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
  return config;
}

OpNode OpNode::create_from_config(const NodeConfig& config) {
  nlohmann::json layer_json = config.get<nlohmann::json>("layer_config");
  LayerConfig layer_cfg = LayerConfig::from_json(layer_json);
  auto layer = LayerFactory::create(layer_cfg);
  std::string uid = config.get<std::string>("uid");
  OpNode node(uid, std::move(layer));
  return node;
}

}  // namespace tnn