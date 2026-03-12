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
#include <mutex>
#include <shared_mutex>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "binary_serializer.hpp"
#include "channels.hpp"
#include "communicator.hpp"
#include "device/device_manager.hpp"
#include "device/iallocator.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/endian.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/ibuffer.hpp"
#include "io_context_pool.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/event.hpp"

namespace tnn {

constexpr uint32_t DEFAULT_IO_THREADS = 4;
constexpr uint32_t DEFAULT_MAX_PACKET_SIZE = 4 * 1024 * 1024 + 64;  // 4MB + header
constexpr uint32_t DEFAULT_SOCKETS_PER_ENDPOINT = 4;

class TCPCommunicator : public Communicator {
private:
  // Hard-coded chunking strategy. Will put into config later if needed.
  void init_peer_ctx(const Endpoint &endpoint, std::shared_ptr<TCPChannel> channel) {
    std::unique_lock<std::shared_mutex> lock(channels_mutex_);
    auto it = peer_ctxs_.find(endpoint);
    if (it == peer_ctxs_.end()) {
      auto slicer = std::make_unique<BlockSlicer>(config_.max_packet_size);
      auto aggregator = std::make_unique<RawAggregator>(int_allocator_);
      PeerContext peer_ctx = make_peer_context(endpoint, std::move(slicer), std::move(aggregator));
      auto [new_it, inserted] = peer_ctxs_.emplace(endpoint, peer_ctx);
      if (!inserted) {
        throw std::runtime_error("Failed to insert new peer context for endpoint: " +
                                 endpoint.id());
      }
      std::cout << "Successfully connected to peer: " << endpoint.id() << std::endl;
      it = new_it;
    }
    channel->set_context(it->second);
    auto &channels = endpoint_channels_[endpoint];
    channels.push_back(channel);
  }

public:
  struct Config {
    uint32_t num_io_threads = DEFAULT_IO_THREADS;
    uint32_t max_packet_size = DEFAULT_MAX_PACKET_SIZE;
    uint32_t skts_per_endpoint = DEFAULT_SOCKETS_PER_ENDPOINT;
  };

  explicit TCPCommunicator(const Endpoint &endpoint, IAllocator &out_allocator,
                           TCPCommunicator::Config config)
      : Communicator(endpoint),
        int_allocator_(PoolAllocator::instance(getHost(), defaultFlowHandle)),
        out_allocator_(out_allocator),
        serializer_(out_allocator),
        config_(config),
        io_context_pool_(config.num_io_threads),
        acceptor_(io_context_pool_.acceptor()) {
    is_running_ = false;
  }

  ~TCPCommunicator() override { stop(); }

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
      std::lock_guard<std::shared_mutex> lock(channels_mutex_);
      endpoint_channels_.clear();
      peer_ctxs_.clear();
    }
    io_context_pool_.stop();
    if (pool_thread_.joinable()) {
      pool_thread_.join();
    }
  }

  void send_impl(Message &&message, const Endpoint &endpoint) override {
    try {
      asio::co_spawn(io_context_pool_.get(), start_send(endpoint, std::move(message)),
                     asio::detached);
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
      for (size_t i = 0; i < config_.skts_per_endpoint; ++i) {
        asio::io_context &io_ctx = io_context_pool_.get();
        auto channel = std::make_shared<TCPChannel>(io_ctx);
        asio::ip::tcp::resolver resolver(io_ctx);
        std::string host = endpoint.get_parameter<std::string>("host");
        int port = endpoint.get_parameter<int>("port");
        auto endpoints = resolver.resolve(host, std::to_string(port));
        asio::steady_timer timer(io_ctx);
        timer.expires_after(std::chrono::seconds(10));
        std::atomic<bool> connected{false};
        std::error_code connect_ec;
        asio::async_connect(channel->socket, endpoints,
                            [&](std::error_code ec, const asio::ip::tcp::endpoint &) {
                              connect_ec = ec;
                              connected.store(true);
                              timer.cancel();
                            });
        timer.async_wait([&](std::error_code ec) {
          if (!ec && !connected.load()) {
            channel->socket.close();
            connect_ec = asio::error::timed_out;
            connected.store(true);
          }
        });
        while (!connected.load()) {
          io_ctx.poll_one();
        }
        if (connect_ec) {
          std::cerr << "Failed to connect to " << host << ":" << port << " - "
                    << connect_ec.message() << std::endl;
          return false;
        }
        std::error_code ec;
        asio::error_code err = channel->socket.set_option(asio::ip::tcp::no_delay(true), ec);
        if (err) {
          std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;
          return false;
        }
        IdentityMessage identity;
        identity.listening_port = endpoint_.get_parameter<int>("port");
        IBuffer identity_buffer(int_allocator_, IdentityMessage::size());
        identity.serialize(identity_buffer);

        asio::write(channel->socket, asio::buffer(identity_buffer.data(), IdentityMessage::size()));

        init_peer_ctx(endpoint, channel);

        asio::co_spawn(
            channel->socket.get_executor(), [this, channel]() { return receive_loop(channel); },
            asio::detached);
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Connection error: " << e.what() << std::endl;
      return false;
    }
    return false;
  }

  bool disconnect_from_endpoint(const Endpoint &endpoint) override {
    std::lock_guard<std::shared_mutex> lock(channels_mutex_);
    auto it = endpoint_channels_.find(endpoint);
    if (it != endpoint_channels_.end()) {
      for (auto &channel : it->second) {
        channel->close();
      }
      endpoint_channels_.erase(it);
    }
    auto peer_it = peer_ctxs_.find(endpoint);
    if (peer_it != peer_ctxs_.end()) {
      peer_ctxs_.erase(peer_it);
    }
    return true;
  }

private:
  struct IdentityMessage {
    int32_t listening_port;
    static constexpr size_t size() { return sizeof(Endianness) + sizeof(listening_port); }
    void serialize(IBuffer &buffer) const {
      size_t offset = 0;
      buffer.write(offset, tnn::sys_endianness);
      buffer.write(offset, listening_port);
    }
    void deserialize(IBuffer &buffer, size_t &offset) {
      Endianness endianess;
      buffer.read(offset, endianess);
      buffer.set_endianess(endianess);
      buffer.read(offset, listening_port);
    }
  };

  IAllocator &int_allocator_;
  IAllocator &out_allocator_;
  BinarySerializer serializer_;
  Config config_;
  IoContextPool io_context_pool_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread pool_thread_;
  std::atomic<bool> is_running_;

  std::unordered_map<Endpoint, PeerContext> peer_ctxs_;
  std::unordered_map<Endpoint, std::vector<std::shared_ptr<TCPChannel>>> endpoint_channels_;
  std::shared_mutex channels_mutex_;

  asio::awaitable<void> listen() {
    while (is_running_.load(std::memory_order_acquire)) {
      try {
        asio::io_context &target_context = io_context_pool_.get();
        auto new_channel = std::make_shared<TCPChannel>(target_context);

        co_await acceptor_.async_accept(new_channel->socket, asio::use_awaitable);

        if (!is_running_.load(std::memory_order_acquire)) co_return;

        asio::co_spawn(
            new_channel->socket.get_executor(),
            [this, new_channel]() { return process_channel(new_channel); }, asio::detached);
      } catch (const std::exception &e) {
        if (!is_running_.load(std::memory_order_acquire)) co_return;
        std::cerr << "Accept error: " << e.what() << std::endl;
      }
    }
  }

  asio::awaitable<void> process_channel(std::shared_ptr<TCPChannel> channel) {
    try {
      std::error_code ec;
      auto err = channel->socket.set_option(asio::ip::tcp::no_delay(true), ec);
      if (ec || err) std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;
      IBuffer identity_buffer(int_allocator_, IdentityMessage::size());
      identity_buffer.resize(IdentityMessage::size());
      co_await asio::async_read(channel->socket,
                                asio::buffer(identity_buffer.data(), IdentityMessage::size()),
                                asio::use_awaitable);
      IdentityMessage identity;
      size_t offset = 0;
      identity.deserialize(identity_buffer, offset);

      auto raw_endpoint = channel->socket.remote_endpoint();
      Endpoint endpoint =
          Endpoint::tcp(raw_endpoint.address().to_string(), identity.listening_port);

      init_peer_ctx(endpoint, channel);

      co_await receive_loop(channel);
    } catch (const std::exception &e) {
      handle_channel_error(channel, asio::error::operation_aborted);
    }
  }

  asio::awaitable<void> receive_loop(std::shared_ptr<TCPChannel> channel) {
    try {
      while (is_running_.load(std::memory_order_acquire)) {
        PacketHeader packet_header;
        auto header_dptr = int_allocator_.allocate(PacketHeader::size());
        auto header_buffer = IBuffer(std::move(header_dptr));
        header_buffer.resize(PacketHeader::size());

        co_await asio::async_read(channel->socket,
                                  asio::buffer(header_buffer.data(), header_buffer.size()),
                                  asio::use_awaitable);

        size_t header_offset = 0;
        serializer_.deserialize(header_buffer, header_offset, packet_header);

        dptr read_buffer = channel->context()->fetch_packet(packet_header);

        auto read_start = Clock::now();
        co_await asio::async_read(channel->socket,
                                  asio::buffer(read_buffer.get(), packet_header.packet_length),
                                  asio::use_awaitable);
        auto read_end = Clock::now();
        GlobalProfiler::add_event(
            {EventType::COMMUNICATION, read_start, read_end, "Packet Read", endpoint_.id()});

        bool is_complete = false;
        is_complete = channel->context()->commit_packet(packet_header);

        if (is_complete) {
          auto deserialize_start = Clock::now();
          dptr data_buffer = channel->context()->finalize(packet_header);
          if (!data_buffer) {
            std::cerr << "Error finalizing message buffer for "
                      << channel->context()->endpoint().id() << std::endl;
            continue;
          }
          auto msg_buffer = IBuffer(std::move(data_buffer));
          msg_buffer.resize(packet_header.msg_length);
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
      handle_channel_error(channel, asio::error::operation_aborted);
    }
  }

  void handle_channel_error(std::shared_ptr<TCPChannel> channel, std::error_code ec) {
    Endpoint endpoint;
    if (channel->context()) {
      endpoint = channel->context()->endpoint();
    }

    if (channel->socket.is_open()) {
      std::error_code close_ec;
      auto err = channel->socket.close(close_ec);
      if (err) {
        std::cerr << "Error closing socket: " << close_ec.message() << std::endl;
      }
    }
    channel->close();

    if (endpoint) {
      std::unique_lock<std::shared_mutex> lock(channels_mutex_);
      auto it = endpoint_channels_.find(endpoint);
      if (it != endpoint_channels_.end()) {
        auto &channels = it->second;
        for (auto chan_it = channels.begin(); chan_it != channels.end();) {
          if (*chan_it == channel) {
            chan_it = channels.erase(chan_it);
          } else {
            ++chan_it;
          }
        }
        if (channels.empty()) {
          endpoint_channels_.erase(it);
          peer_ctxs_.erase(endpoint);
          std::cout << "All channels to " << endpoint.id() << " closed." << std::endl;
        }
      }
    }
  }

  asio::awaitable<void> start_send(Endpoint endpoint, Message message) {
    std::vector<std::shared_ptr<TCPChannel>> channels;
    {
      std::shared_lock<std::shared_mutex> lock(channels_mutex_);
      auto it = endpoint_channels_.find(endpoint);
      if (it != endpoint_channels_.end()) {
        channels = it->second;
      }
    }
    if (channels.empty()) {
      std::cerr << "Error: No active channels for " << endpoint.id() << std::endl;
      co_return;
    }
    auto packets = get_write(channels.front()->context(), message);
    // round-robin packets dispatch
    int conn_index = 0;
    while (!packets.empty()) {
      auto packet = std::move(packets.back());
      packets.pop_back();
      auto channel = channels[conn_index];
      channel->enqueue_write(std::move(packet));
      asio::co_spawn(
          channel->socket.get_executor(),
          [this, channel]() { return write_packets(channel->acquire_write(), channel); },
          asio::detached);

      conn_index = (conn_index + 1) % channels.size();
    }
    co_return;
  }

  std::vector<Packet> get_write(PeerContext peer_ctx, const Message &message) {
    auto serialize_start = Clock::now();
    size_t msg_size = message.size();
    auto data = int_allocator_.allocate(msg_size);
    auto buffer = IBuffer(std::move(data));
    size_t offset = 0;
    serializer_.serialize(buffer, offset, message);
    std::vector<Packet> packets = peer_ctx->slice(std::move(buffer));
    auto serialize_end = Clock::now();
    GlobalProfiler::add_event({EventType::COMMUNICATION, serialize_start, serialize_end,
                               "Message Serialize", peer_ctx->endpoint().id()});
    return packets;
  }

  asio::awaitable<void> write_packets(std::unique_ptr<Channel::WriteHandle> write_handle,
                                      std::shared_ptr<TCPChannel> channel) {
    if (!write_handle) co_return;

    try {
      while (is_running_.load(std::memory_order_acquire)) {
        Packet packet;
        if (!write_handle->queue().try_pop(packet)) break;
        size_t offset = 0;
        auto header_buffer = IBuffer(int_allocator_, PacketHeader::size());
        serializer_.serialize(header_buffer, offset, packet.header);

        std::array<asio::const_buffer, 2> buffers = {
            asio::buffer(header_buffer.data(), header_buffer.size()),
            asio::buffer(packet.data.get(), packet.header.packet_length)};

        auto write_start = Clock::now();
        co_await asio::async_write(channel->socket, buffers, asio::use_awaitable);
        auto write_end = Clock::now();
        GlobalProfiler::add_event({EventType::COMMUNICATION, write_start, write_end, "Packet Write",
                                   channel->context()->endpoint().id()});
      }
    } catch (const std::exception &e) {
      handle_channel_error(channel, asio::error::operation_aborted);
    }
  }
};
}  // namespace tnn