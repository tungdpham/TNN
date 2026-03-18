/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <infiniband/verbs.h>
#include <netinet/in.h>

#include <asio.hpp>
#include <asio/awaitable.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/use_awaitable.hpp>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "common/archiver.hpp"
#include "communicator.hpp"
#include "device/device_manager.hpp"
#include "device/iallocator.hpp"
#include "device/ibv_allocator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/io.hpp"
#include "distributed/roce_channel.hpp"
#include "endpoint.hpp"

namespace tnn {

class RoCECommunicator : public Communicator {
private:
  std::string device_name_;
  int port_;
  asio::io_context io_context_;
  RoCEDevice device_;
  RoCECQ cq_obj_;
  IbvAllocator ibv_allocator_;
  BinarySerializer serializer_;

  asio::ip::tcp::acceptor acceptor_;
  std::thread io_thread_;

  std::atomic<bool> is_running_{false};
  std::atomic<uint64_t> msg_serial_id_counter_{0};

  std::unordered_map<Endpoint, std::shared_ptr<RoCEChannel>> channels_;
  std::unordered_map<uint32_t, std::shared_ptr<RoCEChannel>> qp_map_;
  std::mutex channels_mutex_;

public:
  struct Config {
    uint64_t master_slab_size = 256 * 1024 * 1024;
  };

  RoCECommunicator(const std::string &host, int port, const std::string &device_name, int gid_index,
                   const Config &config)
      : Communicator(Endpoint::roce(host, port, device_name, gid_index)),
        device_name_(device_name),
        port_(port),
        device_(device_name, 1, gid_index),
        cq_obj_(device_, io_context_, ROCE_SQ_DEPTH + ROCE_RQ_DEPTH),
        ibv_allocator_(getHost(), device_.get_pd(), config.master_slab_size),
        serializer_(ibv_allocator_),
        acceptor_(io_context_),
        is_running_(true) {
    asio::co_spawn(
        io_context_,
        [this]() { return cq_obj_.run_loop(is_running_, [this](ibv_wc *wc) { process_wc(wc); }); },
        asio::detached);
  }

  static std::unique_ptr<RoCECommunicator> create(const Endpoint &endpoint,
                                                  RoCECommunicator::Config config) {
    std::string host = endpoint.get_parameter<std::string>("host");
    int port = endpoint.get_parameter<int>("port");
    std::string device_name = endpoint.get_parameter<std::string>("device_name");
    int gid_index = endpoint.get_parameter<int>("gid_index");
    return std::make_unique<RoCECommunicator>(host, port, device_name, gid_index, config);
  }

  ~RoCECommunicator() override { stop(); }

  void stop() {
    is_running_ = false;
    std::error_code ec;
    auto err = acceptor_.close(ec);
    if (err) {
      std::cerr << "Error closing acceptor: " << ec.message() << std::endl;
    }
    io_context_.stop();
    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    {
      std::lock_guard<std::mutex> lock(channels_mutex_);
      for (auto &pair : channels_) {
        pair.second->close();
      }
      channels_.clear();
      qp_map_.clear();
    }
  }

  void start_server() {
    asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port_);
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();
    asio::co_spawn(io_context_, [this]() { return accept_loop(); }, asio::detached);
    io_thread_ = std::thread([this]() { io_context_.run(); });
  }
  void send_impl(Message &&message, const Endpoint &endpoint) override {
    Sizer sizer;
    sizer(message);
    size_t msg_size = sizer.size();
    dptr *data_buffer = new dptr(ibv_allocator_.allocate(msg_size));
    Writer data_writer(*data_buffer);
    data_writer(message);
    uint64_t msg_id = msg_serial_id_counter_.fetch_add(1);
    std::shared_ptr<RoCEChannel> channel;
    {
      std::lock_guard<std::mutex> lock(channels_mutex_);
      auto it = channels_.find(endpoint);
      if (it != channels_.end()) {
        channel = it->second;
      }
    }

    if (!channel || channel->is_closed) {
      if (!connect_to_endpoint(endpoint)) {
        std::cerr << "RoCEChannel connection failed for endpoint: " << endpoint.id() << std::endl;
        delete data_buffer;
        return;
      }
      {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        channel = channels_[endpoint];
      }
    }

    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    if (ibv_query_qp(channel->qp, &attr, IBV_QP_STATE, &init_attr) == 0) {
      if (attr.qp_state != IBV_QPS_RTS) {
        std::cerr << "QP for " << endpoint.id() << " not in RTS state (state: " << attr.qp_state
                  << "), dropping message\n";
        return;
      }
    }

    channel->enqueue_send(msg_id, data_buffer);
  }

  void flush_output_messages() override {
    std::lock_guard<std::mutex> lock(out_message_mutex_);
    while (!out_message_queue_.empty()) {
      auto [message, endpoint] = std::move(out_message_queue_.front());
      send_message(std::move(message), endpoint);
      out_message_queue_.pop();
    }
  }

  IAllocator &out_allocator() override { return ibv_allocator_; }

protected:
  bool connect_to_endpoint(const Endpoint &endpoint) override {
    try {
      // Establish TCP connection for initial handshake and RoCE channel setup
      std::string host = endpoint.get_parameter<std::string>("host");
      int tcp_port = endpoint.get_parameter<int>("port");
      asio::ip::tcp::socket socket(io_context_);
      asio::ip::tcp::resolver resolver(io_context_);
      asio::connect(socket, resolver.resolve(host, std::to_string(tcp_port)));
      nlohmann::json local_endpoint_json = endpoint.to_json();
      std::string local_endpoint_str = local_endpoint_json.dump();
      uint32_t endpoint_len = local_endpoint_str.length();
      asio::write(socket, asio::buffer(&endpoint_len, sizeof(endpoint_len)));
      asio::write(socket, asio::buffer(local_endpoint_str));
      auto channel = std::make_shared<RoCEChannel>(device_, cq_obj_, ibv_allocator_);
      channel->endpoint = endpoint;

      // Send RoCE channel info
      RoCEChannelInfo my_info = channel->get_local_info();
      dptr my_info_buf = ibv_allocator_.allocate(RoCEChannelInfo::size());
      Writer writer(my_info_buf);
      writer(my_info);
      asio::write(socket, asio::buffer(my_info_buf.get(), RoCEChannelInfo::size()));

      // Read peer's RoCE channel info and transition QP to RTS
      dptr peer_info_buf = ibv_allocator_.allocate(RoCEChannelInfo::size());
      asio::read(socket, asio::buffer(peer_info_buf.get(), RoCEChannelInfo::size()));
      Reader reader(peer_info_buf);
      RoCEChannelInfo peer_info;
      reader(peer_info);
      channel->transition_to_rts(peer_info, my_info.psn);
      for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
        dptr *buf = new dptr(ibv_allocator_.allocate(ROCE_BUFFER_SIZE));
        post_recv_buffer(channel->qp, buf);
        channel->recv_buffers.push_back(buf);
      }
      uint8_t ack = 1;
      asio::write(socket, asio::buffer(&ack, 1));
      uint8_t remote_ack = 0;
      asio::read(socket, asio::buffer(&remote_ack, 1));
      if (remote_ack != 1) {
        throw std::runtime_error("Invalid ACK from remote peer");
      }
      {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        qp_map_[channel->qp->qp_num] = channel;
        channels_[endpoint] = channel;
      }
      std::cout << "Successfully established outgoing channel to " << endpoint.id()
                << " (QP: " << channel->qp->qp_num << ")\n";
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Connect error: " << e.what() << std::endl;
      return false;
    }
  }

  bool disconnect_from_endpoint(const Endpoint &endpoint) override {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    auto it = channels_.find(endpoint);
    if (it != channels_.end()) {
      it->second->close();
      qp_map_.erase(it->second->qp->qp_num);
      channels_.erase(it);
      return true;
    }
    return false;
  }

private:
  void post_recv_buffer(ibv_qp *qp, dptr *buf) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)buf->get();
    sge.length = ROCE_BUFFER_SIZE;
    sge.lkey = ibv_allocator_.get_mr_info(*buf).lkey;

    struct ibv_recv_wr wr, *bad_wr = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)buf;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to post recv" << std::endl;
    }
  }

  void process_wc(ibv_wc *wc) {
    if (wc->status != IBV_WC_SUCCESS) {
      if (wc->status != IBV_WC_WR_FLUSH_ERR) {
        std::cerr << "WC Error: " << ibv_wc_status_str(wc->status) << " for QP " << wc->qp_num
                  << " Opcode: " << wc->opcode << std::endl;
      }

      // Cleanup broken channel
      {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        auto it = qp_map_.find(wc->qp_num);
        if (it != qp_map_.end()) {
          auto err_channel = it->second;
          err_channel->close();
          channels_.erase(err_channel->endpoint);
          qp_map_.erase(it);
        }
      }

      // Cleanup stray memory tags
      uint64_t tag = wc->wr_id & 0x3;
      uint64_t ptr = wc->wr_id & ~0x3ULL;
      if (tag == 1)
        delete (WriteContext *)ptr;
      else if (tag == 2)
        delete (dptr *)ptr;
      return;
    }

    std::shared_ptr<RoCEChannel> channel;
    {
      std::lock_guard<std::mutex> lock(channels_mutex_);
      auto it = qp_map_.find(wc->qp_num);
      if (it != qp_map_.end()) channel = it->second;
    }

    if (!channel) {
      std::cerr << "Received packet from unknown QP: " << wc->qp_num << std::endl;
      return;
    }

    if (wc->opcode & IBV_WC_RECV) {
      auto *buf = (dptr *)(wc->wr_id & ~3ULL);

      channel->handle_recv_wc(wc, buf, [this](dptr *complete_buf) {
        Message msg;
        try {
          Reader reader(*complete_buf);
          serializer_.deserialize(reader, msg);
          this->enqueue_input_message(std::move(msg));
        } catch (const std::exception &e) {
          std::cerr << "Deserialization error: " << e.what() << "\n";
        }
        delete complete_buf;
      });

    } else if (wc->opcode == IBV_WC_SEND || wc->opcode == IBV_WC_RDMA_WRITE) {
      channel->handle_send_wc(wc);
    }
  }

  asio::awaitable<void> accept_loop() {
    while (is_running_.load(std::memory_order_acquire)) {
      try {
        auto socket = std::make_shared<asio::ip::tcp::socket>(co_await asio::this_coro::executor);
        co_await acceptor_.async_accept(*socket, asio::use_awaitable);
        asio::co_spawn(
            socket->get_executor(), [this, socket]() { return handle_new_channel(socket); },
            asio::detached);
      } catch (const std::exception &e) {
        if (!is_running_.load(std::memory_order_acquire)) co_return;
        std::cerr << "Accept error: " << e.what() << std::endl;
      }
    }
  }

  asio::awaitable<void> handle_new_channel(std::shared_ptr<asio::ip::tcp::socket> socket) {
    try {
      // Read peer endpoint info
      uint32_t endpoint_len;
      co_await asio::async_read(*socket, asio::buffer(&endpoint_len, sizeof(uint32_t)),
                                asio::use_awaitable);
      std::string peer_endpoint_buf(endpoint_len, '\0');
      co_await asio::async_read(*socket, asio::buffer(peer_endpoint_buf.data(), endpoint_len),
                                asio::use_awaitable);
      nlohmann::json peer_endpoint_json = nlohmann::json::parse(peer_endpoint_buf);
      Endpoint peer_endpoint = Endpoint::from_json(peer_endpoint_json);
      auto channel = std::make_shared<RoCEChannel>(device_, cq_obj_, ibv_allocator_);
      channel->endpoint = peer_endpoint;
      RoCEChannelInfo my_info = channel->get_local_info();

      // Send my RoCE channel info
      dptr my_info_buf = ibv_allocator_.allocate(RoCEChannelInfo::size());
      Writer writer(my_info_buf);
      writer(my_info);
      co_await asio::async_write(*socket, asio::buffer(my_info_buf.get(), RoCEChannelInfo::size()),
                                 asio::use_awaitable);

      // Read peer's RoCE channel info and transition QP to RT
      dptr peer_info_buf = ibv_allocator_.allocate(RoCEChannelInfo::size());
      co_await asio::async_read(*socket, asio::buffer(peer_info_buf.get(), RoCEChannelInfo::size()),
                                asio::use_awaitable);
      Reader reader(peer_info_buf);
      RoCEChannelInfo peer_info;
      reader(peer_info);
      channel->transition_to_rts(peer_info, my_info.psn);

      // Post initial recv buffers and complete handshake
      for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
        dptr *buf = new dptr(ibv_allocator_.allocate(ROCE_BUFFER_SIZE));
        post_recv_buffer(channel->qp, buf);
        channel->recv_buffers.push_back(std::move(buf));
      }
      uint8_t remote_ack = 0;
      co_await asio::async_read(*socket, asio::buffer(&remote_ack, 1), asio::use_awaitable);
      uint8_t ack = 1;
      co_await asio::async_write(*socket, asio::buffer(&ack, 1), asio::use_awaitable);
      {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        qp_map_[channel->qp->qp_num] = channel;
        channels_[channel->endpoint] = channel;
      }
      std::cout << "Successfully established incoming channel from " << channel->endpoint.id()
                << " (QP: " << channel->qp->qp_num << ")\n";
    } catch (const std::exception &e) {
      std::cerr << "Handshake error: " << e.what() << std::endl;
    }
  }
};

}  // namespace tnn
