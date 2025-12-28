/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 *
 * Performance-optimized version
 */
#pragma once

#include "asio/buffer.hpp"
#include "binary_serializer.hpp"
#include "buffer_pool.hpp"
#include "communicator.hpp"
#include "connection.hpp"
#include "connection_group.hpp"
#include "distributed/command_type.hpp"
#include "distributed/fragmenter.hpp"
#include "message.hpp"
#include "packet.hpp"

#include <asio.hpp>
#include <asio/error_code.hpp>
#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <variant>

namespace tnn {

constexpr uint32_t DEFAULT_MAX_PACKET_SIZE = 8 * 1024 * 1024 + 64; // 8MB + header
constexpr uint32_t DEFAULT_SOCKETS_PER_ENDPOINT = 4;

class TcpCommunicator : public Communicator {
public:
  explicit TcpCommunicator(const std::string &id, const Endpoint &endpoint,
                           size_t num_io_threads = 1,
                           uint32_t skts_per_endpoint = DEFAULT_SOCKETS_PER_ENDPOINT,
                           uint32_t max_packet_size = DEFAULT_MAX_PACKET_SIZE)
      : Communicator(id), io_context_(), work_guard_(asio::make_work_guard(io_context_)),
        acceptor_(io_context_), socket_(io_context_),
        num_io_threads_(num_io_threads > 0 ? num_io_threads : 1),
        skts_per_endpoint_(skts_per_endpoint), max_packet_size_(max_packet_size) {
    try {
      host_ = endpoint.get_parameter<std::string>("host");
      port_ = std::stoi(endpoint.get_parameter<std::string>("port"));
    } catch (const std::exception &e) {
      std::cerr << "TcpCommunicator initialization error: " << e.what() << std::endl;
      throw;
    }
    is_running_ = false;
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

    io_threads_.reserve(num_io_threads_);
    for (size_t i = 0; i < num_io_threads_; ++i) {
      io_threads_.emplace_back([this]() { io_context_.run(); });
    }
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
      for (auto &[id, socks] : connection_groups_) {
        for (auto &conn : socks.get_connections()) {
          if (conn->socket.is_open()) {
            std::error_code ec;
            asio::error_code err = conn->socket.close(ec);
            if (err) {
              std::cerr << "Error while disconnecting from " << id << ": " << ec.message()
                        << std::endl;
            }
          }
        }
      }
      connection_groups_.clear();
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

  void send_message(const Message &message) override {
    try {
      std::string recipient_id = message.header().recipient_id;

      size_t msg_size = message.size();

      PooledBuffer data_buffer = BufferPool::instance().get_buffer(msg_size);

      // auto serialize_start = std::chrono::high_resolution_clock::now();
      BinarySerializer::serialize(message, *data_buffer);
      // auto serialize_end = std::chrono::high_resolution_clock::now();
      // auto serialize_duration =
      //     std::chrono::duration_cast<std::chrono::microseconds>(serialize_end - serialize_start);
      // std::cout << "Serialization took " << serialize_duration.count() << " microseconds"
      //           << std::endl;

      uint32_t packets_per_msg =
          static_cast<uint32_t>(std::ceil(static_cast<double>(msg_size) / max_packet_size_));

      std::vector<PacketHeader> headers;

      {
        std::shared_lock<std::shared_mutex> lock(connections_mutex_);
        auto it = connection_groups_.find(recipient_id);
        if (it == connection_groups_.end()) {
          return;
        }
        auto &connection_group = it->second;
        headers = connection_group.get_fragmenter().get_headers(*data_buffer, packets_per_msg);
      }

      async_send_buffer(recipient_id, headers, std::move(data_buffer));
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
      for (size_t i = 0; i < skts_per_endpoint_; ++i) {
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
          return false;
        }

        asio::socket_base::send_buffer_size send_buf_opt(262144);
        err = connection->socket.set_option(send_buf_opt, ec);
        if (err) {
          std::cerr << "Failed to set send buffer size: " << ec.message() << std::endl;
          return false;
        }

        asio::socket_base::receive_buffer_size recv_buf_opt(262144);
        err = connection->socket.set_option(recv_buf_opt, ec);
        if (err) {
          std::cerr << "Failed to set receive buffer size: " << ec.message() << std::endl;
          return false;
        }

        connection->set_peer_id(peer_id);

        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          connection_groups_[peer_id].add_conn(connection);
        }

        start_read(connection);
        handshake(connection, this->id_);
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Connection error: " << e.what() << std::endl;
      return false;
    }
    return false;
  }

  bool disconnect_from_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    auto it = connection_groups_.find(peer_id);
    if (it != connection_groups_.end()) {
      auto &connection_group = it->second;
      for (auto &connection : connection_group.get_connections()) {
        if (connection->socket.is_open()) {
          std::error_code ec;
          auto err = connection->socket.close(ec);
          if (err) {
            std::cerr << "Error while disconnecting from " << peer_id << ": " << ec.message()
                      << std::endl;
          }
        }
      }
      connection_groups_.erase(it);
      return true;
    }
    return false;
  }

private:
  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  asio::ip::tcp::acceptor acceptor_;
  asio::ip::tcp::socket socket_;
  std::vector<std::thread> io_threads_;
  size_t num_io_threads_;
  uint32_t skts_per_endpoint_;
  uint32_t max_packet_size_;

  std::string host_;
  int port_;
  std::atomic<bool> is_running_;

  std::unordered_map<std::string, ConnectionGroup> connection_groups_;
  std::shared_mutex connections_mutex_;

  void accept_connections() {
    if (!is_running_.load(std::memory_order_acquire))
      return;

    std::shared_ptr<Connection> new_connection = std::make_shared<Connection>(io_context_);

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

        new_connection->set_peer_id(temp_id);

        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          connection_groups_[temp_id].add_conn(new_connection);
        }

        start_read(new_connection);
      }

      accept_connections();
    });
  }

  void start_read(std::shared_ptr<Connection> connection) {
    if (!is_running_.load(std::memory_order_acquire))
      return;

    std::string current_id = connection->get_peer_id();

    try {
      const size_t packet_header_size = PacketHeader::size();
      connection->read_buffer = BufferPool::instance().get_buffer(packet_header_size);
      connection->read_buffer->resize(packet_header_size);

      asio::async_read(
          connection->socket, asio::buffer(connection->read_buffer->get(), packet_header_size),
          asio::bind_executor(
              connection->strand, [this, connection](std::error_code ec, std::size_t length) {
                if (!ec && is_running_.load(std::memory_order_acquire)) {
                  if (length != packet_header_size) {
                    std::cerr << "Header fixed part read error: expected " << packet_header_size
                              << " bytes, got " << length << " bytes" << std::endl;
                    return;
                  }
                  PacketHeader packet_header;
                  size_t offset = 0;
                  BinarySerializer::deserialize(*connection->read_buffer, offset, packet_header);

                  connection->read_buffer->set_endianess(packet_header.endianess);
                  read_message(connection, packet_header);
                } else {
                  handle_connection_error(connection, ec);
                }
              }));
    } catch (const std::exception &e) {
      std::cerr << "Start Read error: " << e.what() << std::endl;
      handle_connection_error(connection, asio::error::operation_aborted);
    }
  }

  void read_message(std::shared_ptr<Connection> connection, PacketHeader packet_header) {
    uint64_t msg_serial_id = packet_header.msg_serial_id;
    uint64_t offset = packet_header.packet_offset;
    std::string connection_id = connection->get_peer_id();
    // Get buffer pointer while holding lock, then keep it alive via shared ownership
    uint8_t *buffer_ptr = nullptr;

    {
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      auto it = connection_groups_.find(connection_id);
      if (it == connection_groups_.end()) {
        return;
      }
      auto &connection_group = it->second;
      auto &fragmenter = connection_group.get_fragmenter();

      fragmenter.register_packet(msg_serial_id, packet_header);
      PooledBuffer &buf = fragmenter.get_packet_buffer(msg_serial_id, packet_header);

      buffer_ptr = buf->get() + offset;
    }

    try {
      auto read_start = std::chrono::high_resolution_clock::now();

      asio::async_read(
          connection->socket, asio::buffer(buffer_ptr, packet_header.length),
          asio::bind_executor(
              connection->strand, [this, connection, packet_header, msg_serial_id,
                                   read_start](std::error_code ec, std::size_t length) {
                if (!ec && is_running_.load(std::memory_order_acquire)) {
                  std::string connection_id = connection->get_peer_id();
                  if (length != packet_header.length) {
                    std::cerr << "Incomplete packet read: expected " << packet_header.length
                              << " bytes, got " << length << " bytes" << std::endl;
                    return;
                  }
                  auto read_end = std::chrono::high_resolution_clock::now();
                  auto read_duration =
                      std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start);
                  // std::cout << "Packet read time: " << read_duration.count() << " us" <<
                  // std::endl;

                  // Check if message is complete
                  bool is_complete = false;
                  {
                    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
                    auto it = connection_groups_.find(connection_id);
                    if (it != connection_groups_.end()) {
                      auto &connection_group = it->second;
                      auto &fragmenter = connection_group.get_fragmenter();
                      is_complete = fragmenter.commit_packet(msg_serial_id, packet_header);
                    }
                  }

                  if (is_complete) {
                    handle_message(connection_id, msg_serial_id, std::move(connection));
                    return;
                  }
                  start_read(connection);
                }
              }));
    } catch (const std::exception &e) {
      std::cerr << "Message parsing error: " << e.what() << std::endl;
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      auto it = connection_groups_.find(connection_id);
      if (it != connection_groups_.end()) {
        auto &connection_group = it->second;
        for (auto &conn : connection_group.get_connections()) {
          if (conn->socket.is_open()) {
            std::error_code close_ec;
            auto err = conn->socket.close(close_ec);
            if (err) {
              std::cerr << "Error closing socket for connection " << connection_id << ": "
                        << close_ec.message() << std::endl;
            }
          }
        }
        connection_groups_.erase(it);
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
  void handle_message(const std::string &connection_id, uint64_t msg_serial_id,
                      std::shared_ptr<Connection> connection) {
    try {
      MessageState state;
      {
        std::lock_guard<std::shared_mutex> lock(connections_mutex_);
        auto it = connection_groups_.find(connection_id);
        if (it == connection_groups_.end()) {
          throw std::runtime_error("Connection not found");
        }
        auto &connection_group = it->second;
        auto &fragmenter = connection_group.get_fragmenter();
        state = fragmenter.fetch_complete_message(msg_serial_id);
      }
      Message msg;
      size_t offset = 0;
      BinarySerializer::deserialize(*state.buffer, offset, msg);
      if (msg.header().command_type == CommandType::HANDSHAKE) {
        std::string new_peer_id = msg.header().sender_id;
        std::string old_peer_id = connection->get_peer_id();

        if (old_peer_id == new_peer_id) {
          start_read(connection);
          return;
        }

        std::cout << "Handshake: switching identity from " << old_peer_id << " to " << new_peer_id
                  << std::endl;

        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          connection->set_peer_id(new_peer_id);
          auto old_group_it = connection_groups_.find(old_peer_id);
          if (old_group_it != connection_groups_.end()) {
            auto new_group_it = connection_groups_.find(new_peer_id);
            // if already exists, merge old into new. else rename.
            if (new_group_it != connection_groups_.end()) {
              std::cout << "Merging " << old_peer_id
                        << " connection into existing group: " << new_peer_id << std::endl;
              new_group_it->second.add_conn(connection);
              new_group_it->second.get_fragmenter().merge(
                  std::move(old_group_it->second.get_fragmenter()));
              connection_groups_.erase(old_group_it);
            } else {
              std::cout << "Renaming group " << old_peer_id << " to " << new_peer_id << std::endl;
              auto node_handle = connection_groups_.extract(old_group_it);
              node_handle.key() = new_peer_id;
              connection_groups_.insert(std::move(node_handle));
            }
          }
        }

        // 3. Continue reading using the connection object (which now has the new ID)
        start_read(connection);
      } else {
        this->enqueue_input_message(std::move(msg));
        start_read(connection);
      }
    } catch (const std::exception &e) {
      std::cerr << "Deserialization error: " << e.what() << std::endl;
    }
  }

  void handle_connection_error(std::shared_ptr<Connection> connection, std::error_code ec) {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);

    if (connection->socket.is_open()) {
      std::error_code close_ec;
      auto err = connection->socket.close(close_ec);
      if (err) {
        std::cerr << "Error closing socket for connection " << connection->get_peer_id() << ": "
                  << &connection << ": " << close_ec.message() << std::endl;
      }
    }

    std::cout << "Connection to " << connection->get_peer_id() << " closed." << std::endl;
  }

  void async_send_buffer(const std::string &recipient_id, std::vector<PacketHeader> headers,
                         PooledBuffer &&data_buffer) {
    std::vector<std::shared_ptr<Connection>> connections;

    {
      std::shared_lock<std::shared_mutex> lock(connections_mutex_);
      auto it = connection_groups_.find(recipient_id);
      if (it == connection_groups_.end()) {
        return;
      }
      connections = it->second.get_connections();
    }

    if (connections.empty()) {
      return;
    }

    // round-robin selection of connection
    int conn_index = 0;
    for (auto &header : headers) {
      auto &connection = connections[conn_index];
      asio::post(
          connection->strand, [this, recipient_id, connection, header, data_buffer]() mutable {
            bool write_in_progress = !connection->write_queue.empty();
            connection->write_queue.emplace_back(
                WriteOperation(header, data_buffer->get() + header.packet_offset, data_buffer));
            if (connection->write_queue.size() > 100) {
              std::cerr << "Warning: High number of pending messages ("
                        << connection->write_queue.size() << ") for connection " << recipient_id
                        << std::endl;
            }
            if (!write_in_progress) {
              start_async_write(connection);
            }
          });
      conn_index = (conn_index + 1) % connections.size();
    }
  }

  void start_async_write(std::shared_ptr<Connection> connection) {
    if (connection->write_queue.empty()) {
      return;
    }

    auto &write_queue = connection->write_queue;
    if (write_queue.empty()) {
      std::cerr << "Warning: Attempted to start async write with empty write queue." << std::endl;
      return;
    }
    WriteOperation &current_write = write_queue.front();
    PacketHeader packet_header = current_write.packet_header;
    auto packet_header_buffer = BufferPool::instance().get_buffer(PacketHeader::size());
    BinarySerializer::serialize(packet_header, *packet_header_buffer);
    uint8_t *packet_data = current_write.packet_data;

    std::array<asio::const_buffer, 2> buffers = {
        asio::buffer(packet_header_buffer->get(), packet_header_buffer->size()),
        asio::buffer(packet_data, packet_header.length)};

    asio::async_write(connection->socket, buffers,
                      asio::bind_executor(connection->strand,
                                          [this, connection](std::error_code ec, std::size_t) {
                                            if (ec) {
                                              handle_connection_error(connection, ec);
                                              return;
                                            }

                                            connection->write_queue.pop_front();

                                            if (!connection->write_queue.empty()) {
                                              start_async_write(connection);
                                            }
                                          }));
  }

  void handshake(std::shared_ptr<Connection> connection, std::string identification) {
    Message handshake_msg(MessageHeader{connection->get_peer_id(), CommandType::HANDSHAKE},
                          MessageData(std::monostate{}));
    handshake_msg.header().sender_id = identification;

    size_t msg_size = handshake_msg.size();
    PooledBuffer data_buffer = BufferPool::instance().get_buffer(msg_size);
    BinarySerializer::serialize(handshake_msg, *data_buffer);

    uint32_t packets_per_msg = 1;

    std::vector<PacketHeader> headers;
    {
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      auto it = connection_groups_.find(connection->get_peer_id());
      if (it == connection_groups_.end()) {
        std::cerr << "Connection group not found for handshake with " << connection->get_peer_id()
                  << std::endl;
        return;
      }
      auto &connection_group = it->second;
      headers = connection_group.get_fragmenter().get_headers(*data_buffer, packets_per_msg);
    }

    if (headers.size() != 1) {
      std::cerr << "Handshake message should fit in a single packet." << std::endl;
      return;
    }

    PacketHeader header = headers[0];

    connection->write_queue.emplace_back(WriteOperation(header, data_buffer->get(), data_buffer));

    start_async_write(connection);

    std::cout << "Sent handshake to " << connection->get_peer_id() << std::endl;
  }
};
} // namespace tnn