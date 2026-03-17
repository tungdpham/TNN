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
#include "distributed/packet.hpp"
#include "distributed/roce_channel.hpp"
#include "endpoint.hpp"

namespace tnn {

constexpr size_t ROCE_BUFFER_SIZE = 1 * 1024 * 1024;

class RoCECommunicator : public Communicator {
private:
  std::string device_name_;
  int port_;
  RoCEDevice device_;
  RoCECQ cq_obj_;
  IbvAllocator ibv_allocator_;
  BinarySerializer serializer_;

  asio::io_context io_context_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread io_thread_;

  std::atomic<bool> is_running_{false};
  std::atomic<uint64_t> msg_serial_id_counter_{0};

  std::unordered_map<Endpoint, std::shared_ptr<RoCEChannel>> channels_;
  std::unordered_map<uint32_t, std::shared_ptr<RoCEChannel>> qp_map_;
  std::mutex channels_mutex_;

public:
  struct Config {
    uint64_t master_slab_size = 128 * 1024 * 1024;
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
    if (!channel) {
      std::cerr << "RoCEChannel not found for endpoint: " << endpoint.id() << std::endl;
      return;
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
    {
      std::unique_lock<std::mutex> conn_lock(channel->mutex);
      channel->inflight_cv.wait(conn_lock,
                                [&] { return channel->is_closed || channel->inflight_count < 16; });
      if (channel->is_closed) {
        delete data_buffer;
        return;
      }
      channel->inflight_count++;
      channel->pending_sends[msg_id] = data_buffer;
    }
    PacketHeader header(PacketType::MSG_PREPARE, 0, data_buffer->capacity(), 0, 1);
    sizer.reset();
    sizer(header);
    size_t header_size = sizer.size();
    dptr *send_buf = new dptr(ibv_allocator_.allocate(header_size));
    header.msg_serial_id = msg_id;
    Writer header_writer(*send_buf);
    header_writer(header);
    struct ibv_sge sge;
    sge.addr = (uint64_t)send_buf->get();
    sge.length = sizeof(PacketHeader);
    sge.lkey = ibv_allocator_.get_mr_info(*send_buf).lkey;
    struct ibv_send_wr wr, *bad_wr = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)send_buf;
    wr.opcode = IBV_WR_SEND;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;

    if (ibv_post_send(channel->qp, &wr, &bad_wr)) {
      std::cerr << "Failed to send MSG_PREPARE\n";
      delete send_buf;
      {
        std::lock_guard<std::mutex> conn_lock(channel->mutex);
        channel->pending_sends.erase(msg_id);
      }
      return;
    }
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
      auto channel = std::make_shared<RoCEChannel>(device_, cq_obj_);
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
    struct WriteContext {
      uint64_t msg_serial_id;
    };
    if (wc->status != IBV_WC_SUCCESS) {
      std::cerr << "WC Error: " << ibv_wc_status_str(wc->status) << " for QP " << wc->qp_num
                << " Opcode: " << wc->opcode << std::endl;
      {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        auto it = qp_map_.find(wc->qp_num);
        if (it != qp_map_.end()) {
          auto err_channel = it->second;
          auto conn_endpoint = err_channel->endpoint;
          err_channel->close();
          qp_map_.erase(it);
          channels_.erase(conn_endpoint);
        }
      }
      if (wc->opcode == IBV_WC_RDMA_WRITE && (wc->wr_id & 1)) {
        WriteContext *ctx = (WriteContext *)(wc->wr_id & ~1);
        delete ctx;
      } else {
        if (!(wc->opcode & IBV_WC_RECV)) {
          auto *buf = (dptr *)wc->wr_id;
          delete buf;
        }
      }
      return;
    }

    if (wc->opcode & IBV_WC_RECV) {
      std::shared_ptr<RoCEChannel> channel;
      {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        auto it = qp_map_.find(wc->qp_num);
        if (it != qp_map_.end()) channel = it->second;
      }

      auto *buf = (dptr *)wc->wr_id;

      if (channel) {
        if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          uint32_t imm = ntohl(wc->imm_data);
          dptr *dest_buf;
          {
            std::lock_guard<std::mutex> lock(channel->mutex);
            auto it = channel->pending_receives.find(imm);
            if (it != channel->pending_receives.end()) {
              dest_buf = it->second;
              channel->pending_receives.erase(it);
            } else {
              std::cerr << "No pending receive found for imm ID: " << imm << std::endl;
              dest_buf = nullptr;
            }
          }

          if (dest_buf) {
            Message msg;
            try {
              Reader reader(*dest_buf);
              serializer_.deserialize(reader, msg);
              this->enqueue_input_message(std::move(msg));
            } catch (const std::exception &e) {
              std::cerr << "Deserialization error: " << e.what() << "\n";
            }
            delete dest_buf;
          }

          post_recv_buffer(channel->qp, buf);

        } else {
          PacketHeader header;
          Reader reader(*buf);
          reader(header);

          if (header.type == PacketType::MSG_PREPARE) {
            dptr *dest_buf = new dptr(ibv_allocator_.allocate(header.msg_length));

            {
              std::lock_guard<std::mutex> lock(channel->mutex);
              channel->pending_receives[header.msg_serial_id & 0xFFFFFFFF] = dest_buf;
            }

            auto *send_buf = new dptr(ibv_allocator_.allocate(sizeof(PacketHeader) + 12));

            PacketHeader ready_header(PacketType::MSG_READY_TO_WRITE, 12, 0, 0, 1);
            ready_header.msg_serial_id = header.msg_serial_id;

            Writer writer(*send_buf);
            writer(ready_header);
            uint64_t addr = (uint64_t)dest_buf->get();
            uint32_t rkey = ibv_allocator_.get_mr_info(*dest_buf).rkey;
            writer(addr, rkey);

            struct ibv_sge sge;
            sge.addr = (uint64_t)send_buf->get();
            sge.length = sizeof(PacketHeader) + 12;
            sge.lkey = ibv_allocator_.get_mr_info(*send_buf).lkey;

            struct ibv_send_wr wr, *bad_wr = nullptr;
            std::memset(&wr, 0, sizeof(wr));
            wr.wr_id = (uint64_t)send_buf;
            wr.opcode = IBV_WR_SEND;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.send_flags = IBV_SEND_SIGNALED;

            if (ibv_post_send(channel->qp, &wr, &bad_wr)) {
              std::cerr << "Failed to send MSG_READY\n";
              delete send_buf;
              std::lock_guard<std::mutex> lock(channel->mutex);
              channel->pending_receives.erase(header.msg_serial_id & 0xFFFFFFFF);
            }

          } else if (header.type == PacketType::MSG_READY_TO_WRITE) {
            uint64_t remote_addr;
            uint32_t rkey;
            Reader reader(*buf);
            reader(remote_addr, rkey);

            dptr *source_buf;
            {
              std::lock_guard<std::mutex> lock(channel->mutex);
              auto it = channel->pending_sends.find(header.msg_serial_id);
              if (it != channel->pending_sends.end()) {
                source_buf = it->second;
              } else {
                std::cerr << "No pending send found for MSG_READY with serial ID: "
                          << header.msg_serial_id << std::endl;
                source_buf = nullptr;
              }
            }

            if (source_buf) {
              struct ibv_sge sge;
              sge.addr = (uint64_t)source_buf->get();
              sge.length = source_buf->capacity();
              sge.lkey = ibv_allocator_.get_mr_info(*source_buf).lkey;

              WriteContext *ctx = new WriteContext{header.msg_serial_id};

              struct ibv_send_wr wr, *bad_wr = nullptr;
              std::memset(&wr, 0, sizeof(wr));
              wr.wr_id = (uint64_t)ctx | 1;
              wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
              wr.sg_list = &sge;
              wr.num_sge = 1;
              wr.send_flags = IBV_SEND_SIGNALED;
              wr.imm_data = htonl((uint32_t)(header.msg_serial_id & 0xFFFFFFFF));
              wr.wr.rdma.remote_addr = remote_addr;
              wr.wr.rdma.rkey = rkey;

              if (ibv_post_send(channel->qp, &wr, &bad_wr)) {
                std::cerr << "Failed to post RDMA write\n";
                delete ctx;
              }
            }
          }

          post_recv_buffer(channel->qp, buf);
        }
      } else {
        std::cerr << "Received packet from unknown QP: " << wc->qp_num << std::endl;
      }

    } else if (wc->opcode == IBV_WC_SEND || wc->opcode == IBV_WC_RDMA_WRITE) {
      if (wc->wr_id & 1) {
        WriteContext *ctx = (WriteContext *)(wc->wr_id & ~1);
        std::shared_ptr<RoCEChannel> channel;
        {
          std::lock_guard<std::mutex> lock(channels_mutex_);
          auto it = qp_map_.find(wc->qp_num);
          if (it != qp_map_.end()) channel = it->second;
        }

        if (channel) {
          std::lock_guard<std::mutex> lock(channel->mutex);
          auto send_it = channel->pending_sends.find(ctx->msg_serial_id);
          if (send_it != channel->pending_sends.end()) {
            delete send_it->second;
            channel->pending_sends.erase(send_it);
          }
          channel->inflight_count--;
          channel->inflight_cv.notify_one();
        }
        delete ctx;
      } else {
        auto *buf = (dptr *)wc->wr_id;
        delete buf;
      }
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
      auto channel = std::make_shared<RoCEChannel>(device_, cq_obj_);
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
