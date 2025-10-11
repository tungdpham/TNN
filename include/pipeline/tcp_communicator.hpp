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
#include "message.hpp"
#include "network_serialization.hpp"
#include "tbuffer.hpp"
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

  std::shared_ptr<TBuffer> get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      auto buffer = std::make_shared<TBuffer>(min_size);
      return buffer;
    }

    // Thread-local pool to avoid contention
    thread_local std::deque<std::shared_ptr<TBuffer>> local_pool;

    for (auto it = local_pool.begin(); it != local_pool.end(); ++it) {
      if ((*it)->capacity() >= min_size) {
        auto buffer = std::move(*it);
        local_pool.erase(it);
        buffer->clear();
        return buffer;
      }
    }

    auto buffer = std::make_shared<TBuffer>(min_size);
    return buffer;
  }

  void return_buffer(std::shared_ptr<TBuffer> buffer) {
    if (!buffer || is_shutting_down_.load(std::memory_order_relaxed))
      return;

    thread_local std::deque<std::shared_ptr<TBuffer>> local_pool;

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

class TcpPipelineCommunicator : public PipelineCommunicator {
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

  void send_message(const Message &message) override {
    try {
      // std::cout << "Sending message to " << message.header.recipient_id << " of type "
      //           << static_cast<int>(message.header.command_type) << " size " << message.size()
      //           << " bytes" << std::endl;
      // Pre-allocate buffer with estimated size (header + message)
      size_t msg_size = message.size();

      FixedHeader fixed_header = FixedHeader(msg_size);

      // Allocate buffer for FixedHeader + Message
      auto buffer = std::make_shared<TBuffer>(msg_size + FixedHeader::size());

      BinarySerializer::serialize(fixed_header, *buffer);
      BinarySerializer::serialize(message, *buffer);

      // std::cout << "DEBUG send: buffer size after serialization = " << buffer->size()
      //           << ", expected = " << (msg_size + FixedHeader::size()) << std::endl;

      async_send_buffer(message.header.recipient_id, std::move(buffer));

    } catch (const std::exception &e) {
      std::cerr << "Send error: " << e.what() << std::endl;
    }
  }

  void flush_output_messages() override {
    std::unique_lock<std::mutex> lock(this->out_message_mutex_, std::try_to_lock);

    if (!lock.owns_lock() || this->out_message_queue_.empty()) {
      return;
    }

    while (!this->out_message_queue_.empty()) {
      auto &msg = this->out_message_queue_.front();
      send_message(msg);
      this->out_message_queue_.pop();
    }

    lock.unlock();
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
    std::shared_ptr<TBuffer> buffer;

    explicit WriteOperation(std::shared_ptr<TBuffer> buf) : buffer(std::move(buf)) {}
  };

  struct Connection {
    asio::ip::tcp::socket socket;
    std::shared_ptr<TBuffer> read_buffer;

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

    // read fixed-size header part first
    const size_t fixed_header_size = FixedHeader::size();
    connection->read_buffer->resize(fixed_header_size);
    // No need to fill with zeros, async_read will overwrite

    asio::async_read(connection->socket,
                     asio::buffer(connection->read_buffer->get(), fixed_header_size),
                     [this, connection_id, connection](std::error_code ec, std::size_t length) {
                       if (!ec && is_running_.load(std::memory_order_acquire)) {
                         if (length != fixed_header_size) {
                           std::cerr << "Header fixed part read error: expected "
                                     << fixed_header_size << " bytes, got " << length << " bytes"
                                     << std::endl;
                           return;
                         }
                         read_message(connection_id, connection);
                       } else {
                         handle_connection_error(connection_id, ec);
                       }
                     });
  }

  void read_message(const std::string &connection_id, std::shared_ptr<Connection> connection) {
    TBuffer &buf = *connection->read_buffer;
    size_t offset = 0;

    try {
      FixedHeader fixed_header;
      BinarySerializer::deserialize(buf, offset, fixed_header);

      const size_t fixed_header_size = FixedHeader::size();
      buf.resize(fixed_header.length + fixed_header_size);

      asio::async_read(
          connection->socket, asio::buffer(buf.get() + fixed_header_size, fixed_header.length),
          [this, connection_id, connection, fixed_header](std::error_code ec, std::size_t length) {
            if (!ec && is_running_.load(std::memory_order_acquire)) {
              if (length != fixed_header.length) {
                std::cerr << "Message body read error: expected " << fixed_header.length
                          << " bytes, got " << length << " bytes" << std::endl;
                return;
              }
              handle_message(connection_id, *connection->read_buffer,
                             fixed_header.length + FixedHeader::size());
              start_read(connection_id, connection);
            }
          });

    } catch (const std::exception &e) {
      std::cerr << "Message parsing error: " << e.what() << std::endl;
      // Don't call handle_connection_error with empty error_code as it shows "Success"
      // Just close the connection directly
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      auto it = connections_.find(connection_id);
      if (it != connections_.end()) {
        if (it->second->socket.is_open()) {
          std::error_code close_ec;
          it->second->socket.close(close_ec);
        }
        connections_.erase(it);
      }
    }
  }

  void handle_message(const std::string &connection_id, const TBuffer &buffer, size_t length) {
    try {
      Message message;
      size_t offset = FixedHeader::size(); // Skip the fixed header, already parsed
      BinarySerializer::deserialize(buffer, offset, message);

      // TODO: Set sender_id properly and do validation middlewares.
      message.header.sender_id = connection_id;
      this->enqueue_input_message(message);
    } catch (const std::exception &e) {
      std::cerr << "Deserialization error: " << e.what() << std::endl;
    }
  }

  void handle_connection_error(const std::string &connection_id, std::error_code ec) {
    std::cerr << "Connection " << connection_id << " error: " << ec.message() << std::endl;
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

  void async_send_buffer(const std::string &recipient_id, std::shared_ptr<TBuffer> buffer) {
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

    if (!connection->writing.exchange(true, std::memory_order_acquire)) {
      start_async_write(recipient_id, connection);
    }
  }

  void start_async_write(const std::string &connection_id, std::shared_ptr<Connection> connection) {
    std::shared_ptr<TBuffer> write_buffer;

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
        connection->socket, asio::buffer(write_buffer->get(), write_buffer->size()),
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
  static std::unique_ptr<TcpPipelineCommunicator> create_server(asio::io_context &io_context,
                                                                int listen_port) {
    return std::make_unique<TcpPipelineCommunicator>(io_context, "", listen_port);
  }

  template <typename T = float>
  static std::unique_ptr<TcpPipelineCommunicator> create_client(asio::io_context &io_context) {
    return std::make_unique<TcpPipelineCommunicator>(io_context);
  }
};

} // namespace tpipeline