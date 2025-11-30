/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 *
 * Performance-optimized version
 */
#pragma once

#include "asio/error_code.hpp"
#include "binary_serializer.hpp"
#include "communicator.hpp"
#include "message.hpp"
#include "tbuffer.hpp"

#include <asio.hpp>
#include <atomic>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace tnn {

class BufferPool;

class BufferDeleter {
public:
  explicit BufferDeleter(BufferPool *pool = nullptr) : pool_(pool) {}

  void operator()(TBuffer *ptr) const;

private:
  BufferPool *pool_;
};

using PooledBuffer = std::unique_ptr<TBuffer, BufferDeleter>;

class BufferPool {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;
  static constexpr size_t MAX_POOL_SIZE = 32;

  PooledBuffer get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    BufferDeleter deleter(this);

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      return PooledBuffer(new TBuffer(min_size), deleter);
    }

    thread_local std::deque<TBuffer *> local_pool;

    while (!local_pool.empty()) {
      TBuffer *raw_buf = local_pool.front();
      local_pool.pop_front();

      if (raw_buf->capacity() >= min_size) {
        raw_buf->clear();

        return PooledBuffer(raw_buf, deleter);
      }

      delete raw_buf;
    }

    return PooledBuffer(new TBuffer(min_size), deleter);
  }

  static BufferPool &instance() {
    static BufferPool pool;
    return pool;
  }

  ~BufferPool() { is_shutting_down_.store(true, std::memory_order_release); }

  void return_buffer_internal(TBuffer *buffer) {
    if (buffer == nullptr) {
      return;
    }

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      delete buffer;
      return;
    }

    thread_local std::deque<TBuffer *> local_pool;
    if (local_pool.size() < MAX_POOL_SIZE) {
      buffer->clear();
      local_pool.push_back(buffer);
    } else {
      delete buffer;
    }
  }

  bool is_shutting_down() const { return is_shutting_down_.load(std::memory_order_relaxed); }

private:
  std::atomic<bool> is_shutting_down_{false};
};

inline void BufferDeleter::operator()(TBuffer *ptr) const {
  if (pool_) {
    pool_->return_buffer_internal(ptr);
  } else {
    delete ptr;
  }
}

class TcpCommunicator : public Communicator {
public:
  explicit TcpCommunicator(const Endpoint &endpoint, size_t num_io_threads = 1)
      : io_context_(), work_guard_(asio::make_work_guard(io_context_)), acceptor_(io_context_),
        socket_(io_context_), num_io_threads_(num_io_threads > 0 ? num_io_threads : 1) {
    try {
      host_ = endpoint.get_parameter<std::string>("host");
      port_ = std::stoi(endpoint.get_parameter<std::string>("port"));
    } catch (const std::exception &e) {
      std::cerr << "TcpCommunicator initialization error: " << e.what() << std::endl;
      throw;
    }
    is_running_ = false;
    if (port_ > 0) {
      start_server();
    }

    io_threads_.reserve(num_io_threads_);
    for (size_t i = 0; i < num_io_threads_; ++i) {
      io_threads_.emplace_back([this]() { io_context_.run(); });
    }
  }

  ~TcpCommunicator() override { stop(); }

  void start_server() {
    if (port_ <= 0) {
      throw std::invalid_argument("Listen port must be greater than 0");
    }

    asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), static_cast<asio::ip::port_type>(port_));
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
      asio::error_code err = acceptor_.close(ec);
      if (err) {
        std::cerr << "Error closing acceptor while stopping: " << ec.message() << std::endl;
      }
    }

    {
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      for (auto &[id, conn] : connections_) {
        if (conn->socket.is_open()) {
          std::error_code ec;
          asio::error_code err = conn->socket.close(ec);
          if (err) {
            std::cerr << "Error while disconnecting from " << id << ": " << ec.message()
                      << std::endl;
          }
        }
      }
      connections_.clear();
    }

    work_guard_.reset();
    io_context_.stop();

    for (auto &thread : io_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    io_threads_.clear();
  }

  void send_message(Message &&message) override {
    try {
      size_t msg_size = message.size();

      FixedHeader fixed_header = FixedHeader(msg_size);

      PooledBuffer buffer = BufferPool::instance().get_buffer(msg_size + FixedHeader::size());

      BinarySerializer::serialize(fixed_header, *buffer);
      BinarySerializer::serialize(message, *buffer);

      async_send_buffer(message.header().recipient_id, std::move(buffer));
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
      auto msg = std::move(this->out_message_queue_.front());
      this->out_message_queue_.pop();
      send_message(std::move(msg));
    }

    lock.unlock();
  }

  bool connect_to_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    try {
      auto connection = std::make_shared<Connection>(io_context_);

      asio::ip::tcp::resolver resolver(io_context_);

      std::string host = endpoint.get_parameter<std::string>("host");
      std::string port = endpoint.get_parameter<std::string>("port");
      auto endpoints = resolver.resolve(host, port);

      asio::connect(connection->socket, endpoints);

      std::error_code ec;
      asio::error_code err = connection->socket.set_option(asio::ip::tcp::no_delay(true), ec);
      if (err) {
        std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;
      }

      asio::socket_base::send_buffer_size send_buf_opt(262144);
      err = connection->socket.set_option(send_buf_opt, ec);
      if (err) {
        std::cerr << "Failed to set send buffer size: " << ec.message() << std::endl;
      }

      asio::socket_base::receive_buffer_size recv_buf_opt(262144);
      err = connection->socket.set_option(recv_buf_opt, ec);
      if (err) {
        std::cerr << "Failed to set receive buffer size: " << ec.message() << std::endl;
      }

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

  bool disconnect_from_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(peer_id);
    if (it != connections_.end()) {
      auto &connection = it->second;
      if (connection->socket.is_open()) {
        std::error_code ec;
        auto err = connection->socket.close(ec);
        if (err) {
          std::cerr << "Error while disconnecting from " << peer_id << ": " << ec.message()
                    << std::endl;
        }
      }
      connections_.erase(it);
      return true;
    }
    return false;
  }

private:
  struct WriteOperation {

    PooledBuffer buffer;

    explicit WriteOperation(PooledBuffer buf) : buffer(std::move(buf)) {}
  };

  struct Connection {
    asio::ip::tcp::socket socket;

    PooledBuffer read_buffer;

    std::deque<WriteOperation> write_queue;
    std::mutex write_mutex;
    std::atomic<bool> writing;

    explicit Connection(asio::io_context &io_ctx)
        : socket(io_ctx), read_buffer(BufferPool::instance().get_buffer()), writing(false) {}

    explicit Connection(asio::ip::tcp::socket sock)
        : socket(std::move(sock)), read_buffer(BufferPool::instance().get_buffer()),
          writing(false) {}

    ~Connection() = default;
  };

  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  asio::ip::tcp::acceptor acceptor_;
  asio::ip::tcp::socket socket_;
  std::vector<std::thread> io_threads_;
  size_t num_io_threads_;

  std::string host_;
  int port_;
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
        asio::error_code nodelay_result =
            new_connection->socket.set_option(asio::ip::tcp::no_delay(true), nodelay_ec);
        if (nodelay_result) {
          std::cerr << "Failed to set TCP_NODELAY: " << nodelay_ec.message() << std::endl;
        }

        asio::socket_base::send_buffer_size send_buf_opt(262144);
        asio::error_code send_buf_result =
            new_connection->socket.set_option(send_buf_opt, nodelay_ec);
        if (send_buf_result) {
          std::cerr << "Failed to set send buffer size: " << nodelay_ec.message() << std::endl;
        }

        asio::socket_base::receive_buffer_size recv_buf_opt(262144);
        auto recv_buf_result = new_connection->socket.set_option(recv_buf_opt, nodelay_ec);
        if (recv_buf_result) {
          std::cerr << "Failed to set receive buffer size: " << nodelay_ec.message() << std::endl;
        }

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

    try {

      const size_t fixed_header_size = FixedHeader::size();
      connection->read_buffer->resize(fixed_header_size);

      asio::async_read(
          connection->socket, asio::buffer(connection->read_buffer->get(), fixed_header_size),
          [this, connection_id, connection](std::error_code ec, std::size_t length) {
            if (!ec && is_running_.load(std::memory_order_acquire)) {
              if (length != fixed_header_size) {
                std::cerr << "Header fixed part read error: expected " << fixed_header_size
                          << " bytes, got " << length << " bytes" << std::endl;
                return;
              }
              FixedHeader fixed_header;
              size_t offset = 0;
              BinarySerializer::deserialize(*connection->read_buffer, offset, fixed_header);

              connection->read_buffer->set_endianess(fixed_header.endianess);
              read_message(connection_id, connection, fixed_header);
            } else {
              handle_connection_error(connection_id, ec);
            }
          });
    } catch (const std::exception &e) {
      std::cerr << "Start Read error: " << e.what() << std::endl;
      handle_connection_error(connection_id, asio::error::operation_aborted);
    }
  }

  void read_message(const std::string &connection_id, std::shared_ptr<Connection> connection,
                    FixedHeader fixed_header) {
    try {
      if (fixed_header.length == 0) {
        throw std::runtime_error("Invalid message length: 0");
      }
      TBuffer &buf = *connection->read_buffer;
      const size_t fixed_header_size = FixedHeader::size();
      buf.resize(fixed_header.length + fixed_header_size);

      asio::async_read(
          connection->socket, asio::buffer(buf.get() + fixed_header_size, fixed_header.length),
          [this, connection_id, connection, fixed_header](std::error_code ec, std::size_t length) {
            if (!ec && is_running_.load(std::memory_order_acquire)) {
              if (length != fixed_header.length) {
                throw std::runtime_error("Incomplete message body received");
              }

              handle_message(connection_id, *connection->read_buffer, length);
              start_read(connection_id, connection);
            }
          });

    } catch (const std::exception &e) {
      std::cerr << "Message parsing error: " << e.what() << std::endl;
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      auto it = connections_.find(connection_id);
      if (it != connections_.end()) {
        if (it->second->socket.is_open()) {
          std::error_code close_ec;
          auto err = it->second->socket.close(close_ec);
          if (err) {
            std::cerr << "Error closing socket for connection " << connection_id << ": "
                      << close_ec.message() << std::endl;
          }
        }
        connections_.erase(it);
      }
    }
  }

  /**
   * @brief a fully received message in the buffer.
   * @param connection_id The ID of the connection from which the message was received.
   * @param buffer The buffer containing the serialized message.
   * @param length The length of the message data in the buffer.
   * @note This function assumes that the buffer contains a complete and valid message.
   */
  void handle_message(const std::string &connection_id, TBuffer &buffer, size_t length) {
    try {
      Message message;
      size_t offset = FixedHeader::size();
      BinarySerializer::deserialize(buffer, offset, message);

      message.header().sender_id = connection_id;
      this->enqueue_input_message(std::move(message));
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
        auto err = it->second->socket.close(close_ec);
        if (err) {
          std::cerr << "Error closing socket for connection " << connection_id << ": "
                    << close_ec.message() << std::endl;
        }
      }

      {
        std::lock_guard<std::mutex> write_lock(it->second->write_mutex);
        it->second->write_queue.clear();
        it->second->writing.store(false, std::memory_order_release);
      }
      connections_.erase(it);
    }
  }

  void async_send_buffer(const std::string &recipient_id, PooledBuffer buffer) {
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
    TBuffer *write_buffer_ptr = nullptr;

    {
      std::lock_guard<std::mutex> write_lock(connection->write_mutex);
      if (connection->write_queue.empty()) {
        connection->writing.store(false, std::memory_order_release);
        return;
      }

      write_buffer_ptr = connection->write_queue.front().buffer.get();
    }

    asio::async_write(connection->socket,
                      asio::buffer(write_buffer_ptr->get(), write_buffer_ptr->size()),
                      [this, connection_id, connection](std::error_code ec, std::size_t) {
                        {
                          std::lock_guard<std::mutex> write_lock(connection->write_mutex);
                          if (!connection->write_queue.empty()) {
                            connection->write_queue.pop_front();
                          }
                        }

                        if (ec) {
                          handle_connection_error(connection_id, ec);
                          connection->writing.store(false, std::memory_order_release);
                          return;
                        }

                        start_async_write(connection_id, connection);
                      });
  }
};
} // namespace tnn