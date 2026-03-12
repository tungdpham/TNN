#pragma once

#include <string>

#include "common/config.hpp"

namespace tnn {

using NodeConfig = TConfig;

class INode {
public:
  INode(std::string uid)
      : uid_(uid) {}

  virtual ~INode() = default;

  INode(const INode&) = delete;
  INode& operator=(const INode&) = delete;
  INode(INode&&) = default;
  INode& operator=(INode&&) = default;

  virtual std::string type() const = 0;
  virtual void save_state(std::ofstream& file) = 0;
  virtual NodeConfig get_config() const = 0;

  std::string uid() const { return uid_; }

protected:
  std::string uid_;
};
}  // namespace tnn