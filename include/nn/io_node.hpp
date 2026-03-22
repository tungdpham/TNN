#pragma once

#include "nn/node.hpp"

namespace tnn {

class OpNode;

class IONode : public INode {
public:
  IONode(std::string uid)
      : INode(uid) {}

  std::string type() const override { return "io_node"; }
  void save_state(std::ofstream& file) override {}
  NodeConfig get_config() const override;

  static IONode create_from_config(const NodeConfig& config);
};

}  // namespace tnn