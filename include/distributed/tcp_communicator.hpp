/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "binary_serializer.hpp"
#include "buffer_pool.hpp"
#include "communicator.hpp"
#include "connection.hpp"
#include "connection_group.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/fragmenter.hpp"
#include "io_context_pool.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/event.hpp"

#include <asio.hpp>
#include <asio/error_code.hpp>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

namespace tnn {

constexpr uint32_t DEFAULT_IO_THREADS = 1;
constexpr uint32_t DEFAULT_MAX_PACKET_SIZE = 16 * 1024 * 1024 + 64; // 16MB + header
constexpr uint32_t DEFAULT_SOCKETS_PER_ENDPOINT = 4;

class TcpCommunicator : public Communicator {
public:
  explicit TcpCommunicator(const Endpoint &endpoint, size_t num_io_threads = DEFAULT_IO_THREADS,
                           uint32_t skts_per_endpoint = DEFAULT_SOCKETS_PER_ENDPOINT,
                           uint32_t max_packet_size = DEFAULT_MAX_PACKET_SIZE)
      : Communicator(endpoint), num_io_threads_(num_io_threads > 0 ? num_io_threads : 1),
        io_context_pool_(num_io_threads_), acceptor_(io_context_pool_.get_acceptor_io_context()),
        skts_per_endpoint_(skts_per_endpoint), max_packet_size_(max_packet_size) {
    is_running_ = false;
  }

  ~TcpCommunicator() override { stop(); }

  void start_server() {
    if (endpoint_.get_parameter<int>("port") <= 0) {
      throw std::invalid_argument("Listen port must be greater than 0");
    }

    asio::ip::tcp::endpoint endpoint(
        asio::ip::tcp::v4(),
        static_cast<asio::ip::port_type>(endpoint_.get_parameter<int>("port")));
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();

    is_running_.store(true, std::memory_order_release);
    accept_connections();

    pool_thread_ = std::thread([this]() { io_context_pool_.run(); });
  }

  void stop() {
    std::cout << "Closing communication server" << std::endl;
    if (!is_running_.load(std::memory_order_acquire))
      return;

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
      for (auto &[endpoint, socks] : connection_groups_) {
        for (auto &conn : socks.get_connections()) {
          if (conn->socket.is_open()) {
            std::error_code ec;
            asio::error_code err = conn->socket.close(ec);
            if (err) {
              std::cerr << "Error while disconnecting from " << endpoint.id() << ": "
                        << ec.message() << std::endl;
            }
          }
        }
      }
      connection_groups_.clear();
    }

    io_context_pool_.stop();

    if (pool_thread_.joinable()) {
      pool_thread_.join();
    }
  }

  void set_use_gpu(bool flag) { serializer_.set_use_gpu(flag); }

  void send_impl(Message &&message, const Endpoint &endpoint) override {
    try {
      start_send(endpoint, std::move(message));
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
      auto [msg, endpoint] = std::move(this->out_message_queue_.front());
      this->out_message_queue_.pop();
      send_impl(std::move(msg), endpoint);
    }

    lock.unlock();
  }

  bool connect_to_endpoint(const Endpoint &endpoint) override {
    try {
      for (size_t i = 0; i < skts_per_endpoint_; ++i) {
        asio::io_context &ctx = io_context_pool_.get_io_context();
        auto connection = std::make_shared<Connection>(ctx);

        asio::ip::tcp::resolver resolver(ctx);

        std::string host = endpoint.get_parameter<std::string>("host");
        int port = endpoint.get_parameter<int>("port");
        auto endpoints = resolver.resolve(host, std::to_string(port));

        asio::connect(connection->socket, endpoints);

        std::error_code ec;
        asio::error_code err = connection->socket.set_option(asio::ip::tcp::no_delay(true), ec);
        if (err) {
          std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;
          return false;
        }

        connection->set_peer_endpoint(endpoint);

        // Send identity message with our listening port
        IdentityMessage identity;
        identity.listening_port = endpoint_.get_parameter<int>("port");
        std::vector<uint8_t> identity_buffer;
        identity.serialize(identity_buffer);

        asio::write(connection->socket, asio::buffer(identity_buffer));

        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          connection_groups_[endpoint].add_conn(connection);
        }

        asio::post(connection->socket.get_executor(),
                   [this, connection]() { start_read(connection); });
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Connection error: " << e.what() << std::endl;
      return false;
    }
    return false;
  }

  bool disconnect_from_endpoint(const Endpoint &endpoint) override {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    auto it = connection_groups_.find(endpoint);
    if (it != connection_groups_.end()) {
      auto &connection_group = it->second;
      for (auto &connection : connection_group.get_connections()) {
        if (connection->socket.is_open()) {
          std::error_code ec;
          auto err = connection->socket.close(ec);
          if (err) {
            std::cerr << "Error while disconnecting from " << endpoint.id() << ": " << ec.message()
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
  struct IdentityMessage {
    int32_t listening_port;

    size_t size() const { return sizeof(listening_port); }

    void serialize(std::vector<uint8_t> &buffer) const {
      size_t offset = buffer.size();
      buffer.resize(offset + sizeof(listening_port));
      std::memcpy(buffer.data() + offset, &listening_port, sizeof(listening_port));
    }

    void deserialize(const std::vector<uint8_t> &buffer, size_t &offset) {
      std::memcpy(&listening_port, buffer.data() + offset, sizeof(listening_port));
      offset += sizeof(listening_port);
    }
  };

  size_t num_io_threads_;
  IoContextPool io_context_pool_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread pool_thread_;
  uint32_t skts_per_endpoint_;
  uint32_t max_packet_size_;
  BinarySerializer serializer_;

  std::atomic<bool> is_running_;

  std::unordered_map<Endpoint, ConnectionGroup> connection_groups_;
  std::shared_mutex connections_mutex_;

  void accept_connections() {
    if (!is_running_.load(std::memory_order_acquire))
      return;

    asio::io_context &target_context = io_context_pool_.get_io_context();
    std::shared_ptr<Connection> new_connection = std::make_shared<Connection>(target_context);

    acceptor_.async_accept(new_connection->socket, [this, new_connection](std::error_code ec) {
      if (!ec && is_running_.load(std::memory_order_acquire)) {
        asio::post(new_connection->socket.get_executor(), [this, new_connection]() {
          std::error_code nodelay_ec;
          asio::error_code nodelay_result =
              new_connection->socket.set_option(asio::ip::tcp::no_delay(true), nodelay_ec);
          if (nodelay_result) {
            std::cerr << "Failed to set TCP_NODELAY: " << nodelay_ec.message() << std::endl;
          }

          // Read identity message to get the remote peer's listening port
          std::vector<uint8_t> identity_buffer(sizeof(int32_t));
          std::error_code read_ec;
          asio::read(new_connection->socket, asio::buffer(identity_buffer), read_ec);

          if (read_ec) {
            std::cerr << "Failed to read identity message: " << read_ec.message() << std::endl;
            return;
          }

          IdentityMessage identity;
          size_t offset = 0;
          identity.deserialize(identity_buffer, offset);

          // Use the remote IP with the listening port from the identity message
          auto raw_endpoint = new_connection->socket.remote_endpoint();
          Endpoint endpoint =
              Endpoint::tcp(raw_endpoint.address().to_string(), identity.listening_port);

          new_connection->set_peer_endpoint(endpoint);

          {
            std::lock_guard<std::shared_mutex> lock(connections_mutex_);
            connection_groups_[endpoint].add_conn(new_connection);
          }

          start_read(new_connection);
        });
      }

      accept_connections();
    });
  }

  void start_read(std::shared_ptr<Connection> connection) {
    if (!is_running_.load(std::memory_order_acquire))
      return;

    Endpoint current_endpoint = connection->get_peer_endpoint();

    try {
      const size_t packet_header_size = PacketHeader::size();
      auto buffer = BufferPool::instance().get_buffer(packet_header_size);
      buffer->resize(packet_header_size);

      asio::async_read(connection->socket, asio::buffer(buffer->get(), packet_header_size),
                       [this, connection, buffer](std::error_code ec, std::size_t length) {
                         if (!ec && is_running_.load(std::memory_order_acquire)) {
                           if (length != packet_header_size) {
                             std::cerr << "Header fixed part read error: expected "
                                       << packet_header_size << " bytes, got " << length << " bytes"
                                       << std::endl;
                             return;
                           }

                           PacketHeader packet_header;
                           size_t offset = 0;

                           serializer_.deserialize(*buffer, offset, packet_header);

                           buffer->set_endianess(packet_header.endianess);
                           read_message(connection, packet_header);
                         } else {
                           handle_connection_error(connection, ec);
                         }
                       });
    } catch (const std::exception &e) {
      std::cerr << "Start Read error: " << e.what() << std::endl;
      handle_connection_error(connection, asio::error::operation_aborted);
    }
  }

  void read_message(std::shared_ptr<Connection> connection, PacketHeader packet_header) {
    uint64_t msg_serial_id = packet_header.msg_serial_id;
    uint64_t offset = packet_header.packet_offset;
    // Get buffer pointer while holding lock, then keep it alive via shared ownership
    PooledBuffer buffer;
    Endpoint connection_endpoint;

    auto register_start = Clock::now();
    {
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      connection_endpoint = connection->get_peer_endpoint();
      auto it = connection_groups_.find(connection_endpoint);
      if (it == connection_groups_.end()) {
        std::cerr << "Connection group not found for " << connection_endpoint.id() << " in ("
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;
        return;
      }
      auto &connection_group = it->second;
      auto &fragmenter = connection_group.get_fragmenter();

      fragmenter.register_packet(msg_serial_id, packet_header);
      buffer = fragmenter.get_packet_buffer(msg_serial_id, packet_header);
    }
    auto register_end = Clock::now();
    GlobalProfiler::add_event({
        EventType::COMMUNICATION,
        register_start,
        register_end,
        "Packet Register",
    });

    if (offset + packet_header.length > buffer->size()) {
      std::cerr << "Packet length of " << packet_header.length << " at offset " << offset
                << " exceeds allocated buffer size of " << buffer->size() << std::endl;
      return;
    }

    try {
      auto read_start = Clock::now();
      asio::async_read(
          connection->socket, asio::buffer(buffer->get() + offset, packet_header.length),
          [this, connection, packet_header, buffer, read_start](std::error_code ec,
                                                                std::size_t length) {
            if (!ec && is_running_.load(std::memory_order_acquire)) {
              if (length != packet_header.length) {
                std::cerr << "Incomplete packet read: expected " << packet_header.length
                          << " bytes, got " << length << " bytes" << std::endl;
                return;
              }
              auto read_end = Clock::now();
              GlobalProfiler::add_event(
                  {EventType::COMMUNICATION, read_start, read_end, "Packet Read", endpoint_.id()});

              // Check if message is complete
              bool is_complete = false;
              {
                std::lock_guard<std::shared_mutex> lock(connections_mutex_);
                auto it = connection_groups_.find(connection->get_peer_endpoint());
                if (it != connection_groups_.end()) {
                  auto &connection_group = it->second;
                  auto &fragmenter = connection_group.get_fragmenter();
                  is_complete =
                      fragmenter.commit_packet(packet_header.msg_serial_id, packet_header);
                }
              }

              if (is_complete) {
                handle_message(packet_header.msg_serial_id, std::move(connection));
                return;
              }
              start_read(connection);
            }
          });
    } catch (const std::exception &e) {
      std::cerr << "Message parsing error: " << e.what() << std::endl;
      std::lock_guard<std::shared_mutex> lock(connections_mutex_);
      auto it = connection_groups_.find(connection->get_peer_endpoint());
      if (it != connection_groups_.end()) {
        auto &connection_group = it->second;
        for (auto &conn : connection_group.get_connections()) {
          if (conn->socket.is_open()) {
            std::error_code close_ec;
            auto err = conn->socket.close(close_ec);
            if (err) {
              std::cerr << "Error closing socket for connection "
                        << connection->get_peer_endpoint().id() << ": " << close_ec.message()
                        << std::endl;
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
  void handle_message(uint64_t msg_serial_id, std::shared_ptr<Connection> connection) {
    try {
      auto deserialize_start = Clock::now();
      MessageState state;
      {
        std::lock_guard<std::shared_mutex> lock(connections_mutex_);
        auto it = connection_groups_.find(connection->get_peer_endpoint());
        if (it == connection_groups_.end()) {
          throw std::runtime_error("Connection not found");
        }
        auto &fragmenter = it->second.get_fragmenter();
        state = fragmenter.fetch_complete_message(msg_serial_id);
      }
      Message msg;
      size_t offset = 0;
      serializer_.deserialize(*state.buffer, offset, msg);

      auto deserialize_end = Clock::now();
      GlobalProfiler::add_event({EventType::COMMUNICATION, deserialize_start, deserialize_end,
                                 "Message Deserialize", endpoint_.id()});

      this->enqueue_input_message(std::move(msg));
      start_read(connection);
    } catch (const std::exception &e) {
      std::cerr << "Deserialization error: " << e.what() << std::endl;
      handle_connection_error(connection, std::make_error_code(std::errc::illegal_byte_sequence));
      return;
    }
  }

  void handle_connection_error(std::shared_ptr<Connection> connection, std::error_code ec) {
    if (connection->socket.is_open()) {
      std::error_code close_ec;
      auto err = connection->socket.close(close_ec);
      if (err) {
        std::cerr << "Error closing socket for connection " << connection->get_peer_endpoint().id()
                  << ": " << &connection << ": " << close_ec.message() << std::endl;
      }
    }

    std::lock_guard<std::shared_mutex> lock(connections_mutex_);

    auto it = connection_groups_.find(connection->get_peer_endpoint());
    if (it != connection_groups_.end()) {
      auto &connection_group = it->second;
      connection_group.remove_conn(connection);
      if (connection_group.get_connections().empty()) {
        connection_groups_.erase(it);
        std::cout << "All connections to " << connection->get_peer_endpoint().id() << " closed."
                  << std::endl;
      }
    }
  }

  void start_send(const Endpoint &endpoint, Message &&message) {
    // delegate serialization to ASIO thread
    asio::post(io_context_pool_.get_io_context(), [this, endpoint, message = std::move(message)]() {
      auto write_ops = get_write(endpoint, message);
      std::vector<std::shared_ptr<Connection>> connections;
      {
        std::shared_lock<std::shared_mutex> lock(connections_mutex_);
        auto it = connection_groups_.find(endpoint);
        if (it == connection_groups_.end()) {
          std::cerr << "Error while sending message: Connection group not found for recipient: "
                    << endpoint.id() << std::endl;
          return;
        }
        auto &connection_group = it->second;
        connections = connection_group.get_connections();
      }

      if (connections.empty()) {
        std::cerr << "Error while sending message: No active connections for recipient: "
                  << endpoint.id() << std::endl;
        return;
      }

      // round-robin selection of connections
      int conn_index = 0;
      while (!write_ops.empty()) {
        auto write_op = std::move(write_ops.back());
        write_ops.pop_back();
        auto connection = connections[conn_index];
        connection->enqueue_write(std::move(write_op));
        asio::post(connection->socket.get_executor(), [this, connection]() {
          start_async_write(connection->acquire_write(), connection);
        });
        conn_index = (conn_index + 1) % connections.size();
      }
    });
  }

  std::vector<WriteOperation> get_write(const Endpoint &endpoint, const Message &message) {
    auto serialize_start = Clock::now();
    size_t msg_size = message.size();
    PooledBuffer data_buffer = BufferPool::instance().get_buffer(msg_size);
    size_t offset = 0;
    serializer_.serialize(*data_buffer, offset, message);

    uint32_t packets_per_msg =
        static_cast<uint32_t>(std::ceil(static_cast<double>(msg_size) / max_packet_size_));

    std::vector<PacketHeader> headers;

    {
      std::shared_lock<std::shared_mutex> lock(connections_mutex_);
      auto it = connection_groups_.find(endpoint);
      if (it == connection_groups_.end()) {
        std::cerr
            << "Error while getting write operation: Connection group not found for recipient: "
            << endpoint.id() << std::endl;
        return {};
      }
      auto &connection_group = it->second;
      headers = connection_group.get_fragmenter().get_headers(*data_buffer, packets_per_msg);
    }

    auto serialize_end = Clock::now();
    GlobalProfiler::add_event({EventType::COMMUNICATION, serialize_start, serialize_end,
                               "Message Serialize", endpoint.id()});

    std::vector<WriteOperation> write_ops;
    for (size_t i = 0; i < headers.size(); ++i) {
      auto header = headers[i];
      write_ops.emplace_back(header, data_buffer, header.packet_offset);
    }
    return write_ops;
  }

  void start_async_write(std::unique_ptr<WriteHandle> write_handle,
                         const std::shared_ptr<Connection> connection) {
    if (!write_handle) {
      return;
    }

    WriteOperation current_write;
    bool has_write = write_handle->queue().try_pop(current_write);

    if (!has_write) {
      return;
    }

    PacketHeader packet_header = current_write.packet_header();
    auto packet_header_buffer = BufferPool::instance().get_buffer(PacketHeader::size());
    size_t offset = 0;
    serializer_.serialize(*packet_header_buffer, offset, packet_header);
    uint8_t *packet_data = current_write.packet_data();

    std::array<asio::const_buffer, 2> buffers = {
        asio::buffer(packet_header_buffer->get(), packet_header_buffer->size()),
        asio::buffer(packet_data, packet_header.length)};

    auto write_start = Clock::now();
    asio::async_write(
        connection->socket, buffers,
        [this, connection, packet_header_buffer, current_write = std::move(current_write),
         write_handle = std::move(write_handle),
         write_start](std::error_code ec, std::size_t) mutable {
          if (ec) {
            handle_connection_error(connection, ec);
            return;
          }
          auto write_end = Clock::now();
          GlobalProfiler::add_event({EventType::COMMUNICATION, write_start, write_end,
                                     "Packet Write", connection->get_peer_endpoint().id()});
          start_async_write(std::move(write_handle), connection);
        });
  }
};
} // namespace tnn