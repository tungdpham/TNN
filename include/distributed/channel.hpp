#pragma once

#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "distributed/packet.hpp"
#include "distributed/peer_context.hpp"

namespace tnn {

// thread-safe queue for outgoing packets for a connection
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

class Channel {
public:
  explicit Channel() {}

  virtual ~Channel() = default;

  virtual void close() = 0;

  PeerContext context() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    if (!context_) {
      std::cerr << "Err: No peer context set for this channel" << std::endl;
      return nullptr;
    }
    return context_;
  }

  void set_context(PeerContext context) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_ = context;
  }

  void enqueue_write(Packet &&packet) { write_queue_.enqueue(std::move(packet)); }

  class WriteHandle {
  public:
    WriteHandle(Channel *conn)
        : conn_(conn) {}
    ~WriteHandle() {
      if (conn_) {
        conn_->release_write();
      }
    }
    WriteQueue &queue() { return conn_->write_queue_; }

  private:
    Channel *conn_;
  };

  std::unique_ptr<WriteHandle> acquire_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_writing_) {
      return nullptr;
    }
    is_writing_ = true;
    return std::make_unique<WriteHandle>(this);
  }

private:
  PeerContext context_;
  std::mutex context_mutex_;
  WriteQueue write_queue_;
  bool is_writing_ = false;
  std::mutex write_mutex_;

  void release_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    is_writing_ = false;
  }
};

}  // namespace tnn