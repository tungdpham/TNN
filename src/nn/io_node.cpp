#include "nn/io_node.hpp"

#include "nn/op_node.hpp"

namespace tnn {

NodeConfig IONode::get_config() const {
  NodeConfig config;
  config.type = "io_node";
  config.set("uid", uid_);

  std::vector<std::string> producer_uids;
  for (auto *node : producers_) {
    producer_uids.push_back(node->uid());
  }
  config.set("producers", producer_uids);

  std::vector<std::string> consumer_uids;
  for (auto *node : consumers_) {
    consumer_uids.push_back(node->uid());
  }
  config.set("consumers", consumer_uids);

  return config;
}

IONode IONode::create_from_config(const NodeConfig &config) {
  std::string uid = config.get<std::string>("uid");
  IONode node(uid);
  return node;
}

}  // namespace tnn
