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

  void add_conn(const std::shared_ptr<Connection> &conn) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_.push_back(conn);
  }

  void remove_conn(const std::shared_ptr<Connection> &conn) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_.erase(std::remove(connections_.begin(), connections_.end(), conn),
                       connections_.end());
  }

  std::vector<std::shared_ptr<Connection>> get_connections() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    return connections_;
  }

private:
  std::string id_;
  Fragmenter fragmenter_;
  mutable std::mutex connections_mutex_;
  std::vector<std::shared_ptr<Connection>> connections_;
};
}  // namespace tnn