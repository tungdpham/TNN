#pragma once

#include <asio.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "distributed/endpoint.hpp"
#include "packet.hpp"

namespace tnn {

// Simple thread-unsafe queue for write operations
class WriteQueue {
public:
  bool try_pop(Packet &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    packet = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void enqueue(Packet &&packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(std::move(packet));
  }

private:
  std::queue<Packet> queue_;
  std::mutex mutex_;
};

class Connection;  // forward decl

// RAII for write queue
class WriteHandle {
public:
  WriteHandle(const WriteHandle &) = delete;
  WriteHandle &operator=(const WriteHandle &) = delete;

  WriteHandle(WriteHandle &&other) noexcept
      : conn_(other.conn_) {
    other.conn_ = nullptr;
  }
  WriteHandle &operator=(WriteHandle &&other) noexcept {
    if (this != &other) {
      conn_ = other.conn_;
      other.conn_ = nullptr;
    }
    return *this;
  }

  explicit WriteHandle(Connection *conn)
      : conn_(conn) {}

  ~WriteHandle() {
    if (conn_) {
      release();
    }
  }

  WriteQueue &queue();

private:
  void release();
  Connection *conn_;
};

class Connection {
public:
  asio::ip::tcp::socket socket;

  explicit Connection(asio::io_context &io_ctx)
      : socket(io_ctx) {}

  explicit Connection(asio::ip::tcp::socket sock)
      : socket(std::move(sock)) {}
  ~Connection() = default;

  void set_peer_endpoint(const Endpoint &new_endpoint) {
    std::lock_guard<std::mutex> lock(id_mutex_);
    endpoint_ = new_endpoint;
  }

  const Endpoint &get_peer_endpoint() const {
    std::lock_guard<std::mutex> lock(id_mutex_);
    return endpoint_;
  }

  std::unique_ptr<WriteHandle> acquire_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_writing_) {
      return nullptr;
    }
    is_writing_ = true;
    return std::make_unique<WriteHandle>(this);
  }

  void enqueue_write(Packet &&packet) { write_queue_.enqueue(std::move(packet)); }

private:
  friend class WriteHandle;
  Endpoint endpoint_;
  mutable std::mutex id_mutex_;

  WriteQueue write_queue_;
  bool is_writing_ = false;
  std::mutex write_mutex_;

  void release_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    is_writing_ = false;
  }
};

inline WriteQueue &WriteHandle::queue() { return conn_->write_queue_; }
inline void WriteHandle::release() { conn_->release_write(); }

}  // namespace tnn