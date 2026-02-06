/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <asio.hpp>
#include <asio/awaitable.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/error_code.hpp>
#include <asio/use_awaitable.hpp>
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

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "binary_serializer.hpp"
#include "communicator.hpp"
#include "connection.hpp"
#include "connection_group.hpp"
#include "device/device_manager.hpp"
#include "device/iallocator.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/fragmenter.hpp"
#include "distributed/tbuffer.hpp"
#include "io_context_pool.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/event.hpp"

namespace tnn {

constexpr uint32_t DEFAULT_IO_THREADS = 4;
constexpr uint32_t DEFAULT_MAX_PACKET_SIZE = 4 * 1024 * 1024 + 64;  // 4MB + header
constexpr uint32_t DEFAULT_SOCKETS_PER_ENDPOINT = 4;

class TcpCommunicator;

struct TcpCommunicatorConfig {
  uint32_t num_io_threads = DEFAULT_IO_THREADS;
  uint32_t max_packet_size = DEFAULT_MAX_PACKET_SIZE;
  uint32_t skts_per_endpoint = DEFAULT_SOCKETS_PER_ENDPOINT;
};

class TcpCommunicator : public Communicator {
public:
  explicit TcpCommunicator(const Endpoint &endpoint, IAllocator &out_allocator,
                           TcpCommunicatorConfig config = TcpCommunicatorConfig())
      : Communicator(endpoint),
        int_allocator_(PoolAllocator::instance(getCPU())),
        out_allocator_(out_allocator),
        serializer_(out_allocator),
        num_io_threads_(config.num_io_threads > 0 ? config.num_io_threads : 1),
        io_context_pool_(num_io_threads_),
        acceptor_(io_context_pool_.get_acceptor_io_context()),
        skts_per_endpoint_(config.skts_per_endpoint),
        max_packet_size_(config.max_packet_size) {
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
    asio::co_spawn(acceptor_.get_executor(), [this]() { return listen(); }, asio::detached);
    pool_thread_ = std::thread([this]() { io_context_pool_.run(); });
  }

  void stop() {
    std::cout << "Closing communication server" << std::endl;
    if (!is_running_.load(std::memory_order_acquire)) return;
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
      for (auto &[endpoint, connection_group] : connection_groups_) {
        connection_group.clear();
      }
      connection_groups_.clear();
    }
    io_context_pool_.stop();
    if (pool_thread_.joinable()) {
      pool_thread_.join();
    }
  }

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

  IAllocator &out_allocator() override { return out_allocator_; }

  bool connect_to_endpoint(const Endpoint &endpoint) override {
    try {
      for (size_t i = 0; i < skts_per_endpoint_; ++i) {
        asio::io_context &ctx = io_context_pool_.get_io_context();
        auto connection = std::make_shared<Connection>(ctx);
        asio::ip::tcp::resolver resolver(ctx);
        std::string host = endpoint.get_parameter<std::string>("host");
        int port = endpoint.get_parameter<int>("port");
        auto endpoints = resolver.resolve(host, std::to_string(port));
        asio::steady_timer timer(ctx);
        timer.expires_after(std::chrono::seconds(10));
        std::atomic<bool> connected{false};
        std::error_code connect_ec;
        asio::async_connect(connection->socket, endpoints,
                            [&](std::error_code ec, const asio::ip::tcp::endpoint &) {
                              connect_ec = ec;
                              connected.store(true);
                              timer.cancel();
                            });
        timer.async_wait([&](std::error_code ec) {
          if (!ec && !connected.load()) {
            connection->socket.close();
            connect_ec = asio::error::timed_out;
            connected.store(true);
          }
        });
        while (!connected.load()) {
          ctx.poll_one();
        }
        if (connect_ec) {
          std::cerr << "Failed to connect to " << host << ":" << port << " - "
                    << connect_ec.message() << std::endl;
          return false;
        }
        std::cout << "Successfully connected to " << host << ":" << port << std::endl;
        std::error_code ec;
        asio::error_code err = connection->socket.set_option(asio::ip::tcp::no_delay(true), ec);
        if (err) {
          std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;
          return false;
        }
        connection->set_peer_endpoint(endpoint);
        IdentityMessage identity;
        identity.listening_port = endpoint_.get_parameter<int>("port");
        std::vector<uint8_t> identity_buffer;
        identity.serialize(identity_buffer);

        asio::write(connection->socket, asio::buffer(identity_buffer));

        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          auto it = connection_groups_.find(endpoint);
          if (it == connection_groups_.end()) {
            connection_groups_.emplace(endpoint, ConnectionGroup(int_allocator_));
            it = connection_groups_.find(endpoint);
          }
          it->second.add_conn(connection);
        }

        asio::co_spawn(
            connection->socket.get_executor(),
            [this, connection]() { return receive_loop(connection); }, asio::detached);
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
    if (it == connection_groups_.end()) {
      return false;
    }
    auto &connection_group = it->second;
    connection_group.clear();
    connection_groups_.erase(it);
    return true;
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

  IAllocator &int_allocator_;
  IAllocator &out_allocator_;
  BinarySerializer serializer_;
  size_t num_io_threads_;
  IoContextPool io_context_pool_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread pool_thread_;
  uint32_t skts_per_endpoint_;
  uint32_t max_packet_size_;
  std::atomic<bool> is_running_;
  std::unordered_map<Endpoint, ConnectionGroup> connection_groups_;
  std::shared_mutex connections_mutex_;

  asio::awaitable<void> listen() {
    while (is_running_.load(std::memory_order_acquire)) {
      try {
        asio::io_context &target_context = io_context_pool_.get_io_context();
        auto new_connection = std::make_shared<Connection>(target_context);

        co_await acceptor_.async_accept(new_connection->socket, asio::use_awaitable);

        if (!is_running_.load(std::memory_order_acquire)) co_return;

        asio::co_spawn(
            new_connection->socket.get_executor(),
            [this, new_connection]() { return process_connection(new_connection); },
            asio::detached);
      } catch (const std::exception &e) {
        if (!is_running_.load(std::memory_order_acquire)) co_return;
        std::cerr << "Accept error: " << e.what() << std::endl;
      }
    }
  }

  asio::awaitable<void> process_connection(std::shared_ptr<Connection> connection) {
    try {
      std::error_code ec;
      auto err = connection->socket.set_option(asio::ip::tcp::no_delay(true), ec);
      if (ec || err) std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;
      std::vector<uint8_t> identity_buffer(sizeof(int32_t));
      co_await asio::async_read(connection->socket, asio::buffer(identity_buffer),
                                asio::use_awaitable);

      IdentityMessage identity;
      size_t offset = 0;
      identity.deserialize(identity_buffer, offset);

      auto raw_endpoint = connection->socket.remote_endpoint();
      Endpoint endpoint =
          Endpoint::tcp(raw_endpoint.address().to_string(), identity.listening_port);
      connection->set_peer_endpoint(endpoint);

      {
        std::lock_guard<std::shared_mutex> lock(connections_mutex_);
        auto it = connection_groups_.find(endpoint);
        if (it == connection_groups_.end()) {
          connection_groups_.emplace(endpoint, ConnectionGroup(int_allocator_));
          it = connection_groups_.find(endpoint);
        }
        it->second.add_conn(connection);
      }

      co_await receive_loop(connection);
    } catch (const std::exception &e) {
      handle_connection_error(connection, asio::error::operation_aborted);
    }
  }

  asio::awaitable<void> receive_loop(std::shared_ptr<Connection> connection) {
    try {
      while (is_running_.load(std::memory_order_acquire)) {
        PacketHeader packet_header;
        auto header_dptr = int_allocator_.allocate(PacketHeader::size());
        auto header_buffer = TBuffer(std::move(header_dptr));
        header_buffer.resize(PacketHeader::size());

        co_await asio::async_read(connection->socket,
                                  asio::buffer(header_buffer.data(), header_buffer.size()),
                                  asio::use_awaitable);

        size_t header_offset = 0;
        serializer_.deserialize(header_buffer, header_offset, packet_header);

        TBuffer read_buffer;
        auto register_start = Clock::now();
        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          auto it = connection_groups_.find(connection->get_peer_endpoint());
          if (it == connection_groups_.end()) co_return;
          auto &fragmenter = it->second.get_fragmenter();
          fragmenter.register_packet(packet_header);
          read_buffer = fragmenter.get_packet_buffer(packet_header);
        }
        auto register_end = Clock::now();
        GlobalProfiler::add_event(
            {EventType::COMMUNICATION, register_start, register_end, "Packet Register"});

        auto read_start = Clock::now();
        co_await asio::async_read(connection->socket,
                                  asio::buffer(read_buffer.data(), packet_header.packet_length),
                                  asio::use_awaitable);
        auto read_end = Clock::now();
        GlobalProfiler::add_event(
            {EventType::COMMUNICATION, read_start, read_end, "Packet Read", endpoint_.id()});

        bool is_complete = false;
        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          auto it = connection_groups_.find(connection->get_peer_endpoint());
          if (it != connection_groups_.end()) {
            is_complete = it->second.get_fragmenter().commit_packet(packet_header);
          }
        }

        if (is_complete) {
          auto deserialize_start = Clock::now();
          TBuffer msg_buffer;
          {
            std::lock_guard<std::shared_mutex> lock(connections_mutex_);
            auto it = connection_groups_.find(connection->get_peer_endpoint());
            if (it != connection_groups_.end()) {
              msg_buffer = it->second.get_fragmenter().fetch_complete_message(packet_header);
            }
          }
          Message msg;
          size_t msg_offset = 0;
          serializer_.deserialize(msg_buffer, msg_offset, msg);
          auto deserialize_end = Clock::now();
          GlobalProfiler::add_event({EventType::COMMUNICATION, deserialize_start, deserialize_end,
                                     "Message Deserialize", endpoint_.id()});
          this->enqueue_input_message(std::move(msg));
        }
      }
    } catch (const std::exception &e) {
      handle_connection_error(connection, asio::error::operation_aborted);
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
    asio::co_spawn(
        io_context_pool_.get_io_context(),
        [this, endpoint, message = std::move(message)]() mutable -> asio::awaitable<void> {
          auto packets = get_write(endpoint, message);
          std::vector<std::shared_ptr<Connection>> connections;
          {
            std::shared_lock<std::shared_mutex> lock(connections_mutex_);
            auto it = connection_groups_.find(endpoint);
            if (it == connection_groups_.end()) {
              std::cerr << "Error: Connection group not found for " << endpoint.id() << std::endl;
              co_return;
            }
            connections = it->second.get_connections();
          }

          if (connections.empty()) {
            std::cerr << "Error: No active connections for " << endpoint.id() << std::endl;
            co_return;
          }

          int conn_index = 0;
          while (!packets.empty()) {
            auto packet = std::move(packets.back());
            packets.pop_back();
            auto connection = connections[conn_index];
            connection->enqueue_write(std::move(packet));

            asio::co_spawn(
                connection->socket.get_executor(),
                [this, connection]() mutable {
                  return write_packets(connection->acquire_write(), connection);
                },
                asio::detached);
            conn_index = (conn_index + 1) % connections.size();
          }
          co_return;
        },
        asio::detached);
  }

  std::vector<Packet> get_write(const Endpoint &endpoint, const Message &message) {
    auto serialize_start = Clock::now();
    size_t msg_size = message.size();
    auto data = int_allocator_.allocate(msg_size);
    auto buffer = TBuffer(std::move(data));
    size_t offset = 0;
    serializer_.serialize(buffer, offset, message);
    uint32_t packets_per_msg =
        static_cast<uint32_t>(std::ceil(static_cast<double>(msg_size) / max_packet_size_));
    std::vector<Packet> packets;
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
      packets = connection_group.get_fragmenter().split(std::move(buffer), packets_per_msg);
    }
    auto serialize_end = Clock::now();
    GlobalProfiler::add_event({EventType::COMMUNICATION, serialize_start, serialize_end,
                               "Message Serialize", endpoint.id()});
    return packets;
  }

  asio::awaitable<void> write_packets(std::unique_ptr<WriteHandle> write_handle,
                                      std::shared_ptr<Connection> connection) {
    if (!write_handle) co_return;

    try {
      while (is_running_.load(std::memory_order_acquire)) {
        Packet packet;
        if (!write_handle->queue().try_pop(packet)) break;

        size_t offset = 0;
        auto header_buffer = TBuffer(int_allocator_, PacketHeader::size());
        serializer_.serialize(header_buffer, offset, packet.header);

        std::array<asio::const_buffer, 2> buffers = {
            asio::buffer(header_buffer.data(), header_buffer.size()),
            asio::buffer(packet.data.get(), packet.header.packet_length)};

        auto write_start = Clock::now();
        co_await asio::async_write(connection->socket, buffers, asio::use_awaitable);
        auto write_end = Clock::now();
        GlobalProfiler::add_event({EventType::COMMUNICATION, write_start, write_end, "Packet Write",
                                   connection->get_peer_endpoint().id()});
      }
    } catch (const std::exception &e) {
      handle_connection_error(connection, asio::error::operation_aborted);
    }
  }
};
}  // namespace tnn