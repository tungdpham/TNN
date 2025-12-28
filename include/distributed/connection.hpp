#pragma once

#include "buffer_pool.hpp"
#include "packet.hpp"
#include <asio.hpp>
#include <mutex>
#include <string>

namespace tnn {
struct WriteOperation {
  PacketHeader packet_header;
  uint8_t *packet_data;
  PooledBuffer data; // Keep buffer alive during async write

  explicit WriteOperation(PacketHeader header, uint8_t *data, PooledBuffer buf)
      : packet_header(header), packet_data(data), data(buf) {}
};

struct Connection {
  asio::ip::tcp::socket socket;
  asio::strand<asio::any_io_executor> strand;

  std::deque<WriteOperation> write_queue;

  PooledBuffer read_buffer;

  explicit Connection(asio::io_context &io_ctx)
      : socket(io_ctx), strand(asio::make_strand(io_ctx)) {}

  explicit Connection(asio::ip::tcp::socket sock)
      : socket(std::move(sock)), strand(asio::make_strand(socket.get_executor())) {}
  ~Connection() = default;

  void set_peer_id(const std::string &new_id) {
    std::lock_guard<std::mutex> lock(id_mutex);
    peer_id = new_id;
  }

  std::string get_peer_id() const {
    std::lock_guard<std::mutex> lock(id_mutex);
    return peer_id;
  }

private:
  std::string peer_id;
  mutable std::mutex id_mutex;
};
} // namespace tnn