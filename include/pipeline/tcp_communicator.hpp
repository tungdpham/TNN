/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
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

namespace tpipeline {

class BufferPool {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;
  static constexpr size_t MAX_POOL_SIZE = 32;

  std::shared_ptr<std::vector<uint8_t>> get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    if (is_shutting_down_) {
      auto buffer = std::make_shared<std::vector<uint8_t>>();
      buffer->resize(min_size);
      return buffer;
    }

    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
      if ((*it)->capacity() >= min_size) {
        auto buffer = *it;
        pool_.erase(it);
        buffer->clear();
        buffer->resize(min_size);
        return buffer;
      }
    }

    auto buffer = std::make_shared<std::vector<uint8_t>>();
    buffer->resize(min_size);
    return buffer;
  }

  void return_buffer(std::shared_ptr<std::vector<uint8_t>> buffer) {
    if (!buffer)
      return;
    if (pool_.size() < MAX_POOL_SIZE) {
      buffer->clear();
      pool_.push_back(buffer);
    }
  }

  static BufferPool &instance() {
    static BufferPool pool;
    return pool;
  }

  ~BufferPool() {
    is_shutting_down_ = true;
    pool_.clear();
  }

private:
  std::mutex mutex_;
  std::deque<std::shared_ptr<std::vector<uint8_t>>> pool_;
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

    is_running_ = true;
    accept_connections();
  }

  void stop() {
    is_running_ = false;

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

      auto serialized_payload =
          std::make_shared<std::vector<uint8_t>>(BinarySerializer::serialize_message(message));

      auto header =
          std::make_shared<uint32_t>(htonl(static_cast<uint32_t>(serialized_payload->size())));

      std::vector<std::shared_ptr<void>> data_owners;
      data_owners.push_back(std::static_pointer_cast<void>(header));
      data_owners.push_back(std::static_pointer_cast<void>(serialized_payload));

      std::vector<asio::const_buffer> buffers;
      buffers.push_back(asio::buffer(header.get(), sizeof(uint32_t)));
      buffers.push_back(asio::buffer(serialized_payload->data(), serialized_payload->size()));

      async_send_buffers(message.recipient_id, std::move(buffers), std::move(data_owners));

    } catch (const std::exception &e) {
    }
  }

  void flush_output_messages() override {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);

    if (this->out_message_queue_.empty()) {
      return;
    }

    while (!this->out_message_queue_.empty()) {
      auto &msg = this->out_message_queue_.front();
      send_message(msg);
      this->out_message_queue_.pop();
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

      {
        std::lock_guard<std::shared_mutex> lock(connections_mutex_);
        connections_[peer_id] = connection;
      }

      start_read(peer_id, connection);

      return true;

    } catch (const std::exception &e) {
      return false;
    }
  }

  int get_listen_port() const { return listen_port_; }
  std::string get_local_endpoint() const { return local_endpoint_; }

private:
  struct WriteOperation {
    std::vector<asio::const_buffer> buffers;
    std::vector<std::shared_ptr<void>> data_owners;

    WriteOperation(std::vector<asio::const_buffer> bufs, std::vector<std::shared_ptr<void>> owners)
        : buffers(std::move(bufs)), data_owners(std::move(owners)) {}
  };

  struct Connection {
    asio::ip::tcp::socket socket;
    std::shared_ptr<std::vector<uint8_t>> read_buffer;
    std::mutex write_mutex;
    std::queue<WriteOperation> write_queue;
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
          // Ignore exceptions during shutdown - buffer pool may already be destroyed
          std::cerr << "Warning: Exception caught while returning buffer to pool during Connection "
                       "destruction.\n";
        }
        read_buffer.reset(); // Explicitly reset the shared_ptr
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
    if (!is_running_)
      return;

    auto new_connection = std::make_shared<Connection>(io_context_);

    acceptor_.async_accept(new_connection->socket, [this, new_connection](std::error_code ec) {
      if (!ec && is_running_) {

        std::error_code nodelay_ec;
        new_connection->socket.set_option(asio::ip::tcp::no_delay(true), nodelay_ec);

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
    if (!is_running_)
      return;

    asio::async_read(
        connection->socket, asio::buffer(connection->read_buffer->data(), 4),
        [this, connection_id, connection](std::error_code ec, [[maybe_unused]] std::size_t length) {
          if (!ec && is_running_) {

            uint32_t msg_length;
            std::memcpy(&msg_length, connection->read_buffer->data(), 4);
            msg_length = ntohl(msg_length);

            if (msg_length > 0 && msg_length < 8192 * 8192) {

              if (connection->read_buffer->size() < msg_length) {
                BufferPool::instance().return_buffer(connection->read_buffer);
                connection->read_buffer = BufferPool::instance().get_buffer(msg_length);
              }

              asio::async_read(
                  connection->socket, asio::buffer(connection->read_buffer->data(), msg_length),
                  [this, connection_id, connection, msg_length](std::error_code ec2, std::size_t) {
                    if (!ec2 && is_running_) {
                      handle_message(connection_id, *connection->read_buffer, msg_length);

                      start_read(connection_id, connection);
                    } else {
                      handle_connection_error(connection_id, ec2);
                    }
                  });
            } else {
              std::cerr << "Invalid message length: " << msg_length << std::endl;
              return;
            }
          } else {
            handle_connection_error(connection_id, ec);
          }
        });
  }

  void handle_message(const std::string &connection_id, const std::vector<uint8_t> &buffer,
                      size_t length) {
    try {
      // No copy! Just pass a pointer to the data in the existing buffer.
      Message<T> message = BinarySerializer::deserialize_message<T>(buffer.data(), length);

      this->enqueue_input_message(message);

    } catch (const std::exception &e) {
    }
  }

  void handle_connection_error(const std::string &connection_id, std::error_code ec) {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(connection_id);
    if (it != connections_.end()) {
      if (it->second->socket.is_open()) {
        it->second->socket.close();
      }

      {
        std::lock_guard<std::mutex> write_lock(it->second->write_mutex);
        std::queue<WriteOperation> empty;
        it->second->write_queue.swap(empty);
        it->second->writing = false;
      }
      connections_.erase(it);
    }
  }

  void async_send_buffers(const std::string &recipient_id, std::vector<asio::const_buffer> buffers,
                          std::vector<std::shared_ptr<void>> data_owners) {
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
      connection->write_queue.emplace(std::move(buffers), std::move(data_owners));
    }

    if (!connection->writing.exchange(true)) {
      start_async_write(recipient_id, connection);
    }
  }

  void start_async_write(const std::string &connection_id, std::shared_ptr<Connection> connection) {
    WriteOperation write_op(std::vector<asio::const_buffer>{},
                            std::vector<std::shared_ptr<void>>{});

    {
      std::lock_guard<std::mutex> write_lock(connection->write_mutex);
      if (connection->write_queue.empty()) {
        connection->writing = false;
        return;
      }
      write_op = std::move(connection->write_queue.front());
      connection->write_queue.pop();
    }

    auto write_op_ptr = std::make_shared<WriteOperation>(std::move(write_op));

    asio::async_write(connection->socket, write_op_ptr->buffers,
                      [this, connection_id, connection, write_op_ptr](std::error_code ec,
                                                                      std::size_t bytes_written) {
                        if (ec) {
                          handle_connection_error(connection_id, ec);
                          connection->writing = false;
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