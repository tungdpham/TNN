#include <asio.hpp>

#include "asio/io_context.hpp"
#include "distributed/channel.hpp"

namespace tnn {
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

  asio::ip::tcp::socket socket;

private:
  bool is_closed_;
  std::mutex mutex_;
  std::condition_variable inflight_cv_;
};
}  // namespace tnn