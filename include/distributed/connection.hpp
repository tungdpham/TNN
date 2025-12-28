#include "buffer_pool.hpp"
#include <asio.hpp>

namespace tnn {
struct WriteOperation {

  PooledBuffer buffer;

  explicit WriteOperation(PooledBuffer &&buf) : buffer(std::move(buf)) {}
};

struct Connection {
  asio::ip::tcp::socket socket;
  asio::strand<asio::any_io_executor> strand;

  PooledBuffer read_buffer;

  std::deque<WriteOperation> write_queue;

  explicit Connection(asio::io_context &io_ctx)
      : socket(io_ctx), strand(asio::make_strand(io_ctx)),
        read_buffer(BufferPool::instance().get_buffer()) {}

  explicit Connection(asio::ip::tcp::socket sock)
      : socket(std::move(sock)), strand(asio::make_strand(socket.get_executor())),
        read_buffer(BufferPool::instance().get_buffer()) {}

  ~Connection() = default;
};
} // namespace tnn