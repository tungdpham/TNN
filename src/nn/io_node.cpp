#include "nn/io_node.hpp"

namespace tnn {

NodeConfig IONode::get_config() const {
  NodeConfig config;
  config.type = "io_node";
  config.set("uid", uid_);

  return config;
}

IONode IONode::create_from_config(const NodeConfig &config) {
  std::string uid = config.get<std::string>("uid");
  IONode node(uid);
  return node;
}

}  // namespace tnn
