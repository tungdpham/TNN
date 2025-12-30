#pragma once

#include "connection.hpp"
#include "fragmenter.hpp"

namespace tnn {
// Represents a group of connections to a specific peer
class ConnectionGroup {
public:
  ConnectionGroup() = default;
  ~ConnectionGroup() = default;

  ConnectionGroup(const ConnectionGroup &) = delete;
  ConnectionGroup &operator=(const ConnectionGroup &) = delete;
  ConnectionGroup(ConnectionGroup &&other) {
    id_ = std::move(other.id_);
    fragmenter_ = std::move(other.fragmenter_);
    connections_ = std::move(other.connections_);
  }
  ConnectionGroup &operator=(ConnectionGroup &&other) {
    if (this != &other) {
      id_ = std::move(other.id_);
      fragmenter_ = std::move(other.fragmenter_);
      connections_ = std::move(other.connections_);
    }
    return *this;
  }

  const std::string &get_id() const { return id_; }
  void set_id(const std::string &id) { id_ = id; }

  Fragmenter &get_fragmenter() { return fragmenter_; }
  const Fragmenter &get_fragmenter() const { return fragmenter_; }

  void add_conn(std::shared_ptr<Connection> conn) { connections_.push_back(std::move(conn)); }
  void remove_conn(const std::shared_ptr<Connection> &conn) {
    connections_.erase(std::remove(connections_.begin(), connections_.end(), conn),
                       connections_.end());
  }
  const std::vector<std::shared_ptr<Connection>> &get_connections() const { return connections_; }

private:
  std::string id_;
  Fragmenter fragmenter_;
  std::vector<std::shared_ptr<Connection>> connections_;
};
} // namespace tnn