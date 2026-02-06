#pragma once

#include "connection.hpp"
#include "device/iallocator.hpp"
#include "fragmenter.hpp"

namespace tnn {
// Represents a group of connections to a specific peer
class ConnectionGroup {
public:
  ConnectionGroup(IAllocator &allocator)
      : fragmenter_(allocator) {}
  ~ConnectionGroup() = default;

  ConnectionGroup(const ConnectionGroup &) = delete;
  ConnectionGroup &operator=(const ConnectionGroup &) = delete;
  ConnectionGroup(ConnectionGroup &&other)
      : id_(std::move(other.id_)),
        fragmenter_(std::move(other.fragmenter_)),
        connections_(std::move(other.connections_)) {}

  ConnectionGroup &operator=(ConnectionGroup &&other) = delete;

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

  void clear() {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (auto &conn : connections_) {
      if (conn->socket.is_open()) {
        std::error_code ec;
        auto err = conn->socket.close(ec);
        if (err) {
          std::cerr << "Error while closing connection to " << conn->get_peer_endpoint().id()
                    << ": " << ec.message() << std::endl;
        }
      }
    }
    connections_.clear();
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