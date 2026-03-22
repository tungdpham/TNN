#include <asio.hpp>
#include <queue>

#include "asio/io_context.hpp"
#include "distributed/channel.hpp"

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

class WriteHandle;

class TCPChannel : public Channel {
public:
  TCPChannel(asio::io_context &io_context)
      : socket(io_context),
        is_closed_(false) {}

  ~TCPChannel() override = default;

  void close() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      is_closed_ = true;
    }
    inflight_cv_.notify_all();
    std::error_code ec;
    auto err = socket.close(ec);
    if (err) {
      std::cerr << "Error closing socket for endpoint " << context()->endpoint().id() << ": "
                << ec.message() << std::endl;
    }
  }

  void enqueue_write(Packet &&packet) { write_queue_.enqueue(std::move(packet)); }

  std::unique_ptr<WriteHandle> acquire_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_writing_) {
      return nullptr;
    }
    is_writing_ = true;
    return std::make_unique<WriteHandle>(this);
  }
  asio::ip::tcp::socket socket;

private:
  bool is_closed_;
  std::mutex mutex_;
  std::condition_variable inflight_cv_;

  friend class WriteHandle;
  WriteQueue write_queue_;
  bool is_writing_ = false;
  std::mutex write_mutex_;

  void release_write() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    is_writing_ = false;
  }
};

class WriteHandle {
public:
  WriteHandle(TCPChannel *conn)
      : conn_(conn) {}
  ~WriteHandle() {
    if (conn_) {
      conn_->release_write();
    }
  }
  WriteQueue &queue() { return conn_->write_queue_; }

private:
  TCPChannel *conn_;
};

}  // namespace tnn