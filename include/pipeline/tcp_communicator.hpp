/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 *
 * Performance-optimized version
 */
#pragma once

#include "asio.hpp"
#include "communicator.hpp"
#include "network_serialization.hpp"
#include <atomic>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tpipeline {

// Lock-free buffer pool using thread-local storage
class BufferPool {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;
  static constexpr size_t MAX_POOL_SIZE = 32;

  std::shared_ptr<std::vector<uint8_t>> get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      auto buffer = std::make_shared<std::vector<uint8_t>>();
      buffer->reserve(min_size);
      return buffer;
    }

    // Thread-local pool to avoid contention
    thread_local std::deque<std::shared_ptr<std::vector<uint8_t>>> local_pool;

    for (auto it = local_pool.begin(); it != local_pool.end(); ++it) {
      if ((*it)->capacity() >= min_size) {
        auto buffer = std::move(*it);
        local_pool.erase(it);
        buffer->clear();
        return buffer;
      }
    }

    auto buffer = std::make_shared<std::vector<uint8_t>>();
    buffer->reserve(min_size);
    return buffer;
  }

  void return_buffer(std::shared_ptr<std::vector<uint8_t>> buffer) {
    if (!buffer || is_shutting_down_.load(std::memory_order_relaxed))
      return;

    thread_local std::deque<std::shared_ptr<std::vector<uint8_t>>> local_pool;

    if (local_pool.size() < MAX_POOL_SIZE) {
      buffer->clear();
      local_pool.push_back(std::move(buffer));
    }
  }

  static BufferPool &instance() {
    static BufferPool pool;
    return pool;
  }

  ~BufferPool() { is_shutting_down_.store(true, std::memory_order_release); }

private:
  std::atomic<bool> is_shutting_down_{false};
};

template <typename T = float> class TcpPipelineCommunicator : public PipelineCommunicator<T> {
public:
  explicit TcpPipelineCommunicator(asio::io_context &io_context,
                                   const std::string &local_endpoint = "", int listen_port = 0)
      : io_context_(io_context), acceptor_(io_context), socket_(io_context),
        listen_port_(listen_port), local_endpoint_(local_endpoint), is_running_(false) {

    if (listen_port > 0) {
      start_server();
    }
  }

  ~TcpPipelineCommunicator() override { stop(); }

  void start_server() {
    if (listen_port_ <= 0) {
      throw std::invalid_argument("Listen port must be greater than 0");
    }

    asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(),
                                     static_cast<asio::ip::port_type>(listen_port_));
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();

    is_running_.store(true, std::memory_order_release);
    accept_connections();
  }

  void stop() {
    is_running_.store(false, std::memory_order_release);

    if (acceptor_.is_open()) {
      std::error_code ec;
      acceptor_.close(ec);
    }

    {
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      for (auto &[id, conn] : connections_) {
        if (conn->socket.is_open()) {
          std::error_code ec;
          conn->socket.close(ec);
        }
      }
      connections_.clear();
    }
  }

  void send_message(const Message<T> &message) override {
    try {
      // Serialize once
      auto serialized = BinarySerializer::serialize_message(message);
      uint32_t msg_length = static_cast<uint32_t>(serialized.size());
      uint32_t header = htonl(msg_length);

      // Create single combined buffer to minimize allocations
      auto combined_buffer = std::make_shared<std::vector<uint8_t>>();
      combined_buffer->reserve(sizeof(uint32_t) + serialized.size());

      // Copy header
      combined_buffer->insert(combined_buffer->end(), reinterpret_cast<uint8_t *>(&header),
                              reinterpret_cast<uint8_t *>(&header) + sizeof(uint32_t));

      // Move payload data
      combined_buffer->insert(combined_buffer->end(), std::make_move_iterator(serialized.begin()),
                              std::make_move_iterator(serialized.end()));

      async_send_buffer(message.recipient_id, std::move(combined_buffer));

    } catch (const std::exception &e) {
      std::cerr << "Send error: " << e.what() << std::endl;
    }
  }

  void flush_output_messages() override {
    std::unique_lock<std::mutex> lock(this->out_message_mutex_, std::try_to_lock);

    if (!lock.owns_lock() || this->out_message_queue_.empty()) {
      return;
    }

    // Batch processing - gather messages per recipient
    std::unordered_map<std::string, std::vector<Message<T>>> batched_messages;

    while (!this->out_message_queue_.empty()) {
      auto &msg = this->out_message_queue_.front();
      batched_messages[msg.recipient_id].push_back(std::move(msg));
      this->out_message_queue_.pop();
    }

    lock.unlock();

    // Send batched messages
    for (auto &[recipient_id, messages] : batched_messages) {
      if (messages.size() == 1) {
        send_message(messages[0]);
      } else {
        send_batched_messages(recipient_id, messages);
      }
    }
  }

  bool connect_to_peer(const std::string &peer_id, const std::string &host, int port) {
    try {
      auto connection = std::make_shared<Connection>(io_context_);

      asio::ip::tcp::resolver resolver(io_context_);
      auto endpoints = resolver.resolve(host, std::to_string(port));

      asio::connect(connection->socket, endpoints);

      std::error_code ec;
      connection->socket.set_option(asio::ip::tcp::no_delay(true), ec);

      // Enable socket buffer optimization
      asio::socket_base::send_buffer_size send_buf_opt(262144); // 256KB
      connection->socket.set_option(send_buf_opt, ec);

      asio::socket_base::receive_buffer_size recv_buf_opt(262144);
      connection->socket.set_option(recv_buf_opt, ec);

      {
        std::lock_guard<std::shared_mutex> lock(connections_mutex_);
        connections_[peer_id] = connection;
      }

      start_read(peer_id, connection);

      return true;

    } catch (const std::exception &e) {
      std::cerr << "Connection error: " << e.what() << std::endl;
      return false;
    }
  }

  int get_listen_port() const { return listen_port_; }
  std::string get_local_endpoint() const { return local_endpoint_; }

private:
  struct WriteOperation {
    std::shared_ptr<std::vector<uint8_t>> buffer;

    explicit WriteOperation(std::shared_ptr<std::vector<uint8_t>> buf) : buffer(std::move(buf)) {}
  };

  struct Connection {
    asio::ip::tcp::socket socket;
    std::shared_ptr<std::vector<uint8_t>> read_buffer;

    // Lock-free write queue using atomic flag and single-producer pattern
    std::deque<WriteOperation> write_queue;
    std::mutex write_mutex;
    std::atomic<bool> writing;

    explicit Connection(asio::io_context &io_ctx)
        : socket(io_ctx), read_buffer(BufferPool::instance().get_buffer()), writing(false) {}

    explicit Connection(asio::ip::tcp::socket sock)
        : socket(std::move(sock)), read_buffer(BufferPool::instance().get_buffer()),
          writing(false) {}

    ~Connection() {
      if (read_buffer) {
        try {
          BufferPool::instance().return_buffer(read_buffer);
        } catch (...) {
          // Ignore exceptions during shutdown
        }
        read_buffer.reset();
      }
    }
  };

  asio::io_context &io_context_;
  asio::ip::tcp::acceptor acceptor_;
  asio::ip::tcp::socket socket_;

  int listen_port_;
  std::string local_endpoint_;
  std::atomic<bool> is_running_;

  std::unordered_map<std::string, std::shared_ptr<Connection>> connections_;
  std::shared_mutex connections_mutex_;

  void accept_connections() {
    if (!is_running_.load(std::memory_order_acquire))
      return;

    auto new_connection = std::make_shared<Connection>(io_context_);

    acceptor_.async_accept(new_connection->socket, [this, new_connection](std::error_code ec) {
      if (!ec && is_running_.load(std::memory_order_acquire)) {

        std::error_code nodelay_ec;
        new_connection->socket.set_option(asio::ip::tcp::no_delay(true), nodelay_ec);

        // Optimize socket buffers
        asio::socket_base::send_buffer_size send_buf_opt(262144);
        new_connection->socket.set_option(send_buf_opt, nodelay_ec);

        asio::socket_base::receive_buffer_size recv_buf_opt(262144);
        new_connection->socket.set_option(recv_buf_opt, nodelay_ec);

        auto remote_endpoint = new_connection->socket.remote_endpoint();
        std::string temp_id =
            remote_endpoint.address().to_string() + ":" + std::to_string(remote_endpoint.port());

        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          connections_[temp_id] = new_connection;
        }

        start_read(temp_id, new_connection);
      }

      accept_connections();
    });
  }

  void start_read(const std::string &connection_id, std::shared_ptr<Connection> connection) {
    if (!is_running_.load(std::memory_order_acquire))
      return;

    asio::async_read(
        connection->socket, asio::buffer(connection->read_buffer->data(), 4),
        [this, connection_id, connection](std::error_code ec, [[maybe_unused]] std::size_t length) {
          if (!ec && is_running_.load(std::memory_order_acquire)) {

            uint32_t msg_length;
            std::memcpy(&msg_length, connection->read_buffer->data(), 4);
            msg_length = ntohl(msg_length);

            if (msg_length > 0 && msg_length < 8192 * 8192) {

              if (connection->read_buffer->capacity() < msg_length) {
                BufferPool::instance().return_buffer(connection->read_buffer);
                connection->read_buffer = BufferPool::instance().get_buffer(msg_length);
              } else {
                connection->read_buffer->resize(msg_length);
              }

              asio::async_read(
                  connection->socket, asio::buffer(connection->read_buffer->data(), msg_length),
                  [this, connection_id, connection, msg_length](std::error_code ec2, std::size_t) {
                    if (!ec2 && is_running_.load(std::memory_order_acquire)) {
                      handle_message(connection_id, *connection->read_buffer, msg_length);
                      start_read(connection_id, connection);
                    } else {
                      handle_connection_error(connection_id, ec2);
                    }
                  });
            } else {
              std::cerr << "Invalid message length: " << msg_length << std::endl;
              handle_connection_error(connection_id, std::error_code());
            }
          } else {
            handle_connection_error(connection_id, ec);
          }
        });
  }

  void handle_message(const std::string &connection_id, const std::vector<uint8_t> &buffer,
                      size_t length) {
    try {
      Message<T> message = BinarySerializer::deserialize_message<T>(buffer.data(), length);
      this->enqueue_input_message(message);
    } catch (const std::exception &e) {
      std::cerr << "Deserialization error: " << e.what() << std::endl;
    }
  }

  void handle_connection_error(const std::string &connection_id, std::error_code ec) {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(connection_id);
    if (it != connections_.end()) {
      if (it->second->socket.is_open()) {
        std::error_code close_ec;
        it->second->socket.close(close_ec);
      }

      {
        std::lock_guard<std::mutex> write_lock(it->second->write_mutex);
        it->second->write_queue.clear();
        it->second->writing.store(false, std::memory_order_release);
      }
      connections_.erase(it);
    }
  }

  void send_batched_messages(const std::string &recipient_id,
                             const std::vector<Message<T>> &messages) {
    // Combine multiple messages into single buffer
    size_t total_size = 0;
    std::vector<std::vector<uint8_t>> serialized_msgs;
    serialized_msgs.reserve(messages.size());

    for (const auto &msg : messages) {
      auto serialized = BinarySerializer::serialize_message(msg);
      total_size += sizeof(uint32_t) + serialized.size();
      serialized_msgs.push_back(std::move(serialized));
    }

    auto combined_buffer = std::make_shared<std::vector<uint8_t>>();
    combined_buffer->reserve(total_size);

    for (auto &serialized : serialized_msgs) {
      uint32_t msg_length = htonl(static_cast<uint32_t>(serialized.size()));
      combined_buffer->insert(combined_buffer->end(), reinterpret_cast<uint8_t *>(&msg_length),
                              reinterpret_cast<uint8_t *>(&msg_length) + sizeof(uint32_t));
      combined_buffer->insert(combined_buffer->end(), std::make_move_iterator(serialized.begin()),
                              std::make_move_iterator(serialized.end()));
    }

    async_send_buffer(recipient_id, std::move(combined_buffer));
  }

  void async_send_buffer(const std::string &recipient_id,
                         std::shared_ptr<std::vector<uint8_t>> buffer) {
    std::shared_ptr<Connection> connection;

    {
      std::shared_lock<std::shared_mutex> lock(connections_mutex_);
      auto it = connections_.find(recipient_id);
      if (it == connections_.end()) {
        return;
      }
      connection = it->second;
    }

    {
      std::lock_guard<std::mutex> write_lock(connection->write_mutex);
      connection->write_queue.emplace_back(std::move(buffer));
    }

    // Atomic test-and-set to avoid spurious write initiations
    if (!connection->writing.exchange(true, std::memory_order_acquire)) {
      start_async_write(recipient_id, connection);
    }
  }

  void start_async_write(const std::string &connection_id, std::shared_ptr<Connection> connection) {
    std::shared_ptr<std::vector<uint8_t>> write_buffer;

    {
      std::lock_guard<std::mutex> write_lock(connection->write_mutex);
      if (connection->write_queue.empty()) {
        connection->writing.store(false, std::memory_order_release);
        return;
      }
      write_buffer = std::move(connection->write_queue.front().buffer);
      connection->write_queue.pop_front();
    }

    asio::async_write(
        connection->socket, asio::buffer(write_buffer->data(), write_buffer->size()),
        [this, connection_id, connection, write_buffer](std::error_code ec, std::size_t) {
          if (ec) {
            handle_connection_error(connection_id, ec);
            connection->writing.store(false, std::memory_order_release);
            return;
          }

          start_async_write(connection_id, connection);
        });
  }
};

class TcpCommunicatorFactory {
public:
  template <typename T = float>
  static std::unique_ptr<TcpPipelineCommunicator<T>> create_server(asio::io_context &io_context,
                                                                   int listen_port) {
    return std::make_unique<TcpPipelineCommunicator<T>>(io_context, "", listen_port);
  }

  template <typename T = float>
  static std::unique_ptr<TcpPipelineCommunicator<T>> create_client(asio::io_context &io_context) {
    return std::make_unique<TcpPipelineCommunicator<T>>(io_context);
  }
};

} // namespace tpipeline