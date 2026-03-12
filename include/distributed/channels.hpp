#pragma once

#include <infiniband/verbs.h>

#include <asio.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "asio/io_context.hpp"
#include "distributed/packet.hpp"
#include "distributed/peer_context.hpp"
#include "distributed/roce_buffer.hpp"

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
    if (!context_) {
      std::cerr << "Err: No peer context set for this channel" << std::endl;
      return nullptr;
    }
    return context_;
  }

  void set_context(PeerContext context) { context_ = context; }

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
  WriteQueue write_queue_;
  bool is_writing_ = false;
  std::mutex write_mutex_;

  void release_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    is_writing_ = false;
  }
};

class TCPChannel : public Channel {
public:
  TCPChannel(asio::io_context &io_context)
      : socket(io_context) {}

  ~TCPChannel() override = default;

  void close() override {
    std::error_code ec;
    auto err = socket.close(ec);
    if (err) {
      std::cerr << "Error closing socket for endpoint " << context()->endpoint().id() << ": "
                << ec.message() << std::endl;
    }
  }

  asio::ip::tcp::socket socket;
};

class RoCEChannel : public Channel {
public:
  ibv_qp *qp = nullptr;
  Endpoint endpoint;
  uint32_t psn = 0;
  std::vector<std::unique_ptr<RoCEBuffer>> recv_buffers;

  std::mutex mutex;
  std::unordered_map<uint64_t, std::shared_ptr<RoCEBuffer>> pending_sends;
  std::unordered_map<uint32_t, std::shared_ptr<RoCEBuffer>> pending_receives;

  ~RoCEChannel() {
    if (qp) {
      ibv_destroy_qp(qp);
      qp = nullptr;
    }
  }

  void close() override {
    std::lock_guard<std::mutex> lock(mutex);
    pending_sends.clear();
    pending_receives.clear();
  }
};

}  // namespace tnn