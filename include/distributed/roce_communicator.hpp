/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>

#include <asio.hpp>
#include <asio/awaitable.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/posix/stream_descriptor.hpp>
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
#include <vector>

#include "communicator.hpp"
#include "device/device_manager.hpp"
#include "device/iallocator.hpp"
#include "device/ibv_allocator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/channels.hpp"
#include "distributed/endian.hpp"
#include "distributed/ibuffer.hpp"
#include "distributed/packet.hpp"
#include "distributed/roce_buffer.hpp"
#include "endpoint.hpp"

namespace tnn {

constexpr int ROCE_SQ_DEPTH = 32;
constexpr int ROCE_RQ_DEPTH = 32;
constexpr size_t ROCE_BUFFER_SIZE = 1 * 1024 * 1024;

struct RoCEChannelInfo {
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
  union ibv_gid gid;

  static constexpr size_t size() {
    return sizeof(Endianness) + sizeof(lid) + sizeof(qpn) + sizeof(psn) + sizeof(gid);
  }
};

class RoCECommunicator : public Communicator {
private:
  std::string device_name_;
  int port_;
  int gid_index_;
  int ib_port_ = 1;

  std::unique_ptr<IbvAllocator> ibv_allocator_;
  IAllocator &out_allocator_;
  BinarySerializer serializer_;
  asio::io_context io_context_;
  asio::ip::tcp::acceptor acceptor_;
  asio::posix::stream_descriptor desc_;
  std::thread io_thread_;

  ibv_context *context_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_cq *cq_ = nullptr;
  ibv_comp_channel *comp_channel_ = nullptr;

  std::atomic<bool> is_running_{false};
  std::atomic<uint64_t> msg_serial_id_counter_{0};

  std::unordered_map<Endpoint, std::shared_ptr<RoCEChannel>> channels_;
  std::unordered_map<uint32_t, std::shared_ptr<RoCEChannel>> qp_map_;
  std::mutex channels_mutex_;

public:
  struct Config {
    uint64_t master_slab_size = 128 * 1024 * 1024;
  };

  explicit RoCECommunicator(const Endpoint &endpoint, IAllocator &out_allocator,
                            RoCECommunicator::Config config)
      : Communicator(endpoint),
        out_allocator_(out_allocator),
        serializer_(out_allocator),
        acceptor_(io_context_),
        desc_(io_context_) {
    device_name_ = endpoint.get_parameter<std::string>("device_name");
    port_ = endpoint.get_parameter<int>("port");
    gid_index_ = endpoint.get_parameter<int>("gid_index");
    init_rdma();
    print_gid_table();
    ibv_allocator_ = std::make_unique<IbvAllocator>(getHost(), pd_, config.master_slab_size);
    is_running_ = true;
    asio::co_spawn(io_context_, [this]() { return cq_event_loop(); }, asio::detached);
  }

  ~RoCECommunicator() override { stop(); }

  void stop() {
    is_running_ = false;
    desc_.cancel();
    io_context_.stop();
    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    {
      std::lock_guard<std::mutex> lock(channels_mutex_);
      channels_.clear();
      qp_map_.clear();
    }

    if (cq_) ibv_destroy_cq(cq_);
    if (comp_channel_) ibv_destroy_comp_channel(comp_channel_);
    if (pd_) ibv_dealloc_pd(pd_);
    if (context_) ibv_close_device(context_);
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
    size_t msg_size = message.size();
    auto data_buffer = std::make_shared<RoCEBuffer>(*ibv_allocator_, msg_size);
    size_t offset = 0;
    serializer_.serialize(*data_buffer, offset, message);
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
      std::lock_guard<std::mutex> conn_lock(channel->mutex);
      channel->pending_sends[msg_id] = data_buffer;
    }
    auto *send_buf = new RoCEBuffer(*ibv_allocator_, PacketHeader::size());
    PacketHeader header(PacketType::MSG_PREPARE, 0, data_buffer->size(), 0, 1);
    header.msg_serial_id = msg_id;
    size_t header_offset = 0;
    serializer_.serialize(*send_buf, header_offset, header);
    struct ibv_sge sge;
    sge.addr = (uint64_t)send_buf->data();
    sge.length = sizeof(PacketHeader);
    sge.lkey = send_buf->get_lkey();
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

  IAllocator &out_allocator() override { return out_allocator_; }

protected:
  bool connect_to_endpoint(const Endpoint &endpoint) override {
    try {
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
      auto channel = std::make_shared<RoCEChannel>();
      channel->endpoint = endpoint;
      channel->qp = create_qp();
      RoCEChannelInfo my_info = get_local_info(channel->qp);
      IBuffer my_info_buf(out_allocator_, RoCEChannelInfo::size());
      serialize_info(my_info, my_info_buf);
      asio::write(socket, asio::buffer(my_info_buf.data(), my_info_buf.size()));
      IBuffer peer_info_buf(out_allocator_, RoCEChannelInfo::size());
      asio::read(socket, asio::buffer(peer_info_buf.data(), peer_info_buf.size()));
      RoCEChannelInfo peer_info = deserialize_info(peer_info_buf);
      modify_qp_to_rts(channel->qp, peer_info, my_info.psn);
      for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
        auto buf = std::make_unique<RoCEBuffer>(*ibv_allocator_, ROCE_BUFFER_SIZE);
        buf->resize(ROCE_BUFFER_SIZE);
        post_recv_buffer(channel->qp, buf.get());
        channel->recv_buffers.push_back(std::move(buf));
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
      qp_map_.erase(it->second->qp->qp_num);
      channels_.erase(it);
      return true;
    }
    return false;
  }

private:
  void serialize_info(const RoCEChannelInfo &info, IBuffer &buffer) {
    size_t offset = 0;
    buffer.write(offset, tnn::sys_endianness);
    buffer.write(offset, info.lid);
    buffer.write(offset, info.qpn);
    buffer.write(offset, info.psn);
    buffer.write(offset, info.gid.raw, 16);
  }

  RoCEChannelInfo deserialize_info(const IBuffer &buffer) {
    Endianness endianness;
    size_t offset = 0;
    buffer.read(offset, endianness);
    RoCEChannelInfo info;
    buffer.read(offset, info.lid);
    buffer.read(offset, info.qpn);
    buffer.read(offset, info.psn);
    buffer.read(offset, info.gid.raw, 16);
    return info;
  }

  void print_gid_table() {
    struct ibv_port_attr port_attr;
    if (ibv_query_port(context_, ib_port_, &port_attr) == 0) {
      std::cout << "[RoCE] GID Table for device " << device_name_ << ":\n";
      for (int i = 0; i < port_attr.gid_tbl_len; ++i) {
        union ibv_gid gid;
        if (ibv_query_gid(context_, ib_port_, i, &gid) == 0) {
          bool empty = true;
          for (int b = 0; b < 16; ++b)
            if (gid.raw[b] != 0) empty = false;
          if (empty) continue;

          std::cout << "  GID Index " << i << ": ";
          auto old_flags = std::cout.flags();
          for (int b = 0; b < 16; ++b) std::cout << (int)gid.raw[b] << (b < 15 ? ":" : "");
          std::cout.flags(old_flags);
          std::cout << "\n";
        }
      }
    }
  }

  void init_rdma() {
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) throw std::runtime_error("Failed to get IB devices list");

    struct ibv_device *ib_dev = nullptr;
    for (int i = 0; i < num_devices; ++i) {
      if (device_name_ == ibv_get_device_name(dev_list[i])) {
        ib_dev = dev_list[i];
        break;
      }
    }

    if (!ib_dev) {
      ibv_free_device_list(dev_list);
      throw std::runtime_error("Device not found: " + device_name_);
    }

    context_ = ibv_open_device(ib_dev);
    ibv_free_device_list(dev_list);
    if (!context_) throw std::runtime_error("Failed to open device");
    pd_ = ibv_alloc_pd(context_);
    if (!pd_) throw std::runtime_error("Failed to alloc PD");
    comp_channel_ = ibv_create_comp_channel(context_);
    if (!comp_channel_) throw std::runtime_error("Failed to create completion channel");
    cq_ = ibv_create_cq(context_, ROCE_SQ_DEPTH + ROCE_RQ_DEPTH, nullptr, comp_channel_, 0);
    if (!cq_) throw std::runtime_error("Failed to create CQ");
    int flags = fcntl(comp_channel_->fd, F_GETFL);
    fcntl(comp_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    desc_.assign(comp_channel_->fd);
    ibv_req_notify_cq(cq_, 0);
    if (gid_index_ == -1) {
      struct ibv_port_attr port_attr;
      int best_gid_index = -1;
      bool found_ipv4 = false;

      if (ibv_query_port(context_, ib_port_, &port_attr) == 0) {
        for (int i = 0; i < port_attr.gid_tbl_len; ++i) {
          union ibv_gid gid;
          if (ibv_query_gid(context_, ib_port_, i, &gid) == 0) {
            bool empty = true;
            for (int b = 0; b < 16; ++b) {
              if (gid.raw[b] != 0) {
                empty = false;
                break;
              }
            }
            if (empty) continue;
            bool is_ipv4 = true;
            for (int b = 0; b < 10; ++b) {
              if (gid.raw[b] != 0) {
                is_ipv4 = false;
                break;
              }
            }
            if (is_ipv4 && gid.raw[10] == 0xff && gid.raw[11] == 0xff) {
              best_gid_index = i;
              found_ipv4 = true;
              break;
            }
            if (best_gid_index == -1) {
              best_gid_index = i;
            }
          }
        }
      }
      gid_index_ = best_gid_index;
      if (gid_index_ != -1) {
        std::cout << "[RoCE] Auto-selected GID Index: " << gid_index_
                  << (found_ipv4 ? " (IPv4/RoCEv2)" : "") << "\n";
      } else {
        throw std::runtime_error(
            "Auto-selection of GID Index failed: No valid GID found on device " + device_name_);
      }
    }
  }

  ibv_qp *create_qp() {
    struct ibv_qp_init_attr init_attr;
    std::memset(&init_attr, 0, sizeof(init_attr));
    init_attr.send_cq = cq_;
    init_attr.recv_cq = cq_;
    init_attr.cap.max_send_wr = ROCE_SQ_DEPTH;
    init_attr.cap.max_recv_wr = ROCE_RQ_DEPTH;
    init_attr.cap.max_send_sge = 2;
    init_attr.cap.max_recv_sge = 1;
    init_attr.qp_type = IBV_QPT_RC;
    ibv_qp *qp = ibv_create_qp(pd_, &init_attr);
    if (!qp) throw std::runtime_error("Failed to create QP");
    return qp;
  }

  RoCEChannelInfo get_local_info(ibv_qp *qp) {
    RoCEChannelInfo info;
    info.qpn = qp->qp_num;
    info.psn = lrand48() & 0xffffff;
    struct ibv_port_attr attr;
    ibv_query_port(context_, ib_port_, &attr);
    info.lid = attr.lid;
    ibv_query_gid(context_, ib_port_, gid_index_, &info.gid);
    return info;
  }

  void modify_qp_to_rts(ibv_qp *qp, const RoCEChannelInfo &peer_info, uint32_t local_psn) {
    struct ibv_qp_attr attr;
    int flags;
    int ret;
    struct ibv_port_attr port_attr;
    if ((ret = ibv_query_port(context_, ib_port_, &port_attr)) != 0) {
      throw std::runtime_error("Failed to query port: " + std::string(std::strerror(ret)));
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ib_port_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if ((ret = ibv_modify_qp(qp, &attr, flags)) != 0) {
      std::cerr << "Failed to modify QP to INIT. ret=" << ret << " (" << std::strerror(ret)
                << ")\n";
      throw std::runtime_error("Failed to modify QP to INIT");
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = (port_attr.active_mtu < IBV_MTU_1024) ? port_attr.active_mtu : IBV_MTU_1024;
    attr.dest_qp_num = peer_info.qpn;
    attr.rq_psn = peer_info.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.dlid = peer_info.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = ib_port_;
    attr.ah_attr.grh.dgid = peer_info.gid;
    attr.ah_attr.grh.sgid_index = gid_index_;
    attr.ah_attr.grh.hop_limit = 64;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if ((ret = ibv_modify_qp(qp, &attr, flags)) != 0) {
      std::cerr << "Failed to modify QP to RTR. ret=" << ret << " (" << std::strerror(ret) << ")\n";
      std::cerr << "  GID Index: " << gid_index_ << "\n";
      std::cerr << "  Remote QPN: " << peer_info.qpn << "\n";
      std::cerr << "  Remote LID: " << peer_info.lid << "\n";
      std::cerr << "  MTU (Active/Path): " << port_attr.active_mtu << "/" << attr.path_mtu << "\n";
      std::cerr << "  Remote GID: ";
      auto old_flags = std::cerr.flags();
      for (int i = 0; i < 16; ++i) std::cerr << (int)peer_info.gid.raw[i] << ":";
      std::cerr.flags(old_flags);
      std::cerr << "\n";

      union ibv_gid local_gid;
      if (ibv_query_gid(context_, ib_port_, gid_index_, &local_gid) == 0) {
        std::cerr << "  Local GID (Index " << gid_index_ << "): ";
        for (int i = 0; i < 16; ++i) std::cerr << (int)local_gid.raw[i] << ":";
        std::cerr.flags(old_flags);
        std::cerr << "\n";

        bool remote_ipv4 = true;
        for (int i = 0; i < 10; ++i)
          if (peer_info.gid.raw[i] != 0) remote_ipv4 = false;
        if (peer_info.gid.raw[10] != 0xff || peer_info.gid.raw[11] != 0xff) remote_ipv4 = false;

        bool local_ipv4 = true;
        for (int i = 0; i < 10; ++i)
          if (local_gid.raw[i] != 0) local_ipv4 = false;
        if (local_gid.raw[10] != 0xff || local_gid.raw[11] != 0xff) local_ipv4 = false;

        if (remote_ipv4 != local_ipv4) {
          std::cerr << "Hint: GID Type mismatch detected!\n"
                    << "      Remote is " << (remote_ipv4 ? "IPv4-mapped" : "IPv6/Link-local")
                    << "\n"
                    << "      Local is " << (local_ipv4 ? "IPv4-mapped" : "IPv6/Link-local") << "\n"
                    << "      Try using a different --gid-index that matches the IP version.\n";
        }
      }

      throw std::runtime_error("Failed to modify QP to RTR");
    }

    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 20;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
            IBV_QP_MAX_QP_RD_ATOMIC;
    if ((ret = ibv_modify_qp(qp, &attr, flags)) != 0) {
      std::cerr << "Failed to modify QP to RTS. ret=" << ret << " (" << std::strerror(ret) << ")\n";
      throw std::runtime_error("Failed to modify QP to RTS");
    }
  }

  void post_recv_buffer(ibv_qp *qp, RoCEBuffer *buf) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)buf->data();
    sge.length = ROCE_BUFFER_SIZE;
    sge.lkey = buf->get_lkey();

    struct ibv_recv_wr wr, *bad_wr = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)buf;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to post recv" << std::endl;
    }
  }

  asio::awaitable<void> cq_event_loop() {
    while (is_running_) {
      try {
        co_await desc_.async_wait(asio::posix::stream_descriptor::wait_read, asio::use_awaitable);

        if (!is_running_) break;

        struct ibv_cq *ev_cq;
        void *ev_ctx;

        if (ibv_get_cq_event(comp_channel_, &ev_cq, &ev_ctx) == 0) {
          ibv_ack_cq_events(ev_cq, 1);

          if (ibv_req_notify_cq(ev_cq, 0)) {
            std::cerr << "Failed to request notify CQ" << std::endl;
          }

          process_completions();
        }
      } catch (const std::exception &e) {
        if (!is_running_) break;
        std::cerr << "Error in CQ event loop: " << e.what() << std::endl;
      }
    }
  }

  void process_completions() {
    struct WriteContext {
      uint64_t msg_serial_id;
    };
    struct ibv_wc wc[16];
    int n;
    while ((n = ibv_poll_cq(cq_, 16, wc)) > 0) {
      for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::cerr << "WC Error: " << ibv_wc_status_str(wc[i].status) << " for QP " << wc[i].qp_num
                    << " Opcode: " << wc[i].opcode << std::endl;
          {
            std::lock_guard<std::mutex> lock(channels_mutex_);
            auto it = qp_map_.find(wc[i].qp_num);
            if (it != qp_map_.end()) {
              auto conn_endpoint = it->second->endpoint;
              qp_map_.erase(it);
              channels_.erase(conn_endpoint);
            }
          }
          if (wc[i].opcode == IBV_WC_RDMA_WRITE && (wc[i].wr_id & 1)) {
            WriteContext *ctx = (WriteContext *)(wc[i].wr_id & ~1);
            delete ctx;
          }
          continue;
        }

        if (wc[i].opcode & IBV_WC_RECV) {
          std::shared_ptr<RoCEChannel> channel;
          {
            std::lock_guard<std::mutex> lock(channels_mutex_);
            auto it = qp_map_.find(wc[i].qp_num);
            if (it != qp_map_.end()) channel = it->second;
          }

          auto *buf = (RoCEBuffer *)wc[i].wr_id;

          if (channel) {
            if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              uint32_t imm = ntohl(wc[i].imm_data);
              std::shared_ptr<RoCEBuffer> dest_buf;
              {
                std::lock_guard<std::mutex> lock(channel->mutex);
                auto it = channel->pending_receives.find(imm);
                if (it != channel->pending_receives.end()) {
                  dest_buf = it->second;
                  channel->pending_receives.erase(it);
                }
              }

              if (dest_buf) {
                Message msg;
                size_t offset = 0;
                try {
                  serializer_.deserialize(*dest_buf, offset, msg);
                  this->enqueue_input_message(std::move(msg));
                } catch (const std::exception &e) {
                  std::cerr << "Deserialization error: " << e.what() << "\n";
                }
              }

              post_recv_buffer(channel->qp, buf);

            } else {
              PacketHeader header;
              size_t offset = 0;
              serializer_.deserialize(*buf, offset, header);

              if (header.type == PacketType::MSG_PREPARE) {
                auto dest_buf = std::make_shared<RoCEBuffer>(*ibv_allocator_, header.msg_length);
                dest_buf->resize(header.msg_length);

                {
                  std::lock_guard<std::mutex> lock(channel->mutex);
                  channel->pending_receives[header.msg_serial_id & 0xFFFFFFFF] = dest_buf;
                }

                auto *send_buf = new RoCEBuffer(*ibv_allocator_, sizeof(PacketHeader) + 12);

                PacketHeader ready_header(PacketType::MSG_READY_TO_WRITE, 12, 0, 0, 1);
                ready_header.msg_serial_id = header.msg_serial_id;

                size_t ready_offset = 0;
                serializer_.serialize(*send_buf, ready_offset, ready_header);
                uint64_t addr = (uint64_t)dest_buf->data();
                uint32_t rkey = dest_buf->get_rkey();
                send_buf->write(ready_offset, addr);
                send_buf->write(ready_offset, rkey);

                struct ibv_sge sge;
                sge.addr = (uint64_t)send_buf->data();
                sge.length = sizeof(PacketHeader) + 12;
                sge.lkey = send_buf->get_lkey();

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
                buf->read(offset, remote_addr);
                buf->read(offset, rkey);

                std::shared_ptr<RoCEBuffer> source_buf;
                {
                  std::lock_guard<std::mutex> lock(channel->mutex);
                  auto it = channel->pending_sends.find(header.msg_serial_id);
                  if (it != channel->pending_sends.end()) source_buf = it->second;
                }

                if (source_buf) {
                  struct ibv_sge sge;
                  sge.addr = (uint64_t)source_buf->data();
                  sge.length = source_buf->size();
                  sge.lkey = source_buf->get_lkey();

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
            std::cerr << "Received packet from unknown QP: " << wc[i].qp_num << std::endl;
          }

        } else if (wc[i].opcode == IBV_WC_SEND || wc[i].opcode == IBV_WC_RDMA_WRITE) {
          if (wc[i].wr_id & 1) {
            WriteContext *ctx = (WriteContext *)(wc[i].wr_id & ~1);
            std::shared_ptr<RoCEChannel> channel;
            {
              std::lock_guard<std::mutex> lock(channels_mutex_);
              auto it = qp_map_.find(wc[i].qp_num);
              if (it != qp_map_.end()) channel = it->second;
            }

            if (channel) {
              std::lock_guard<std::mutex> lock(channel->mutex);
              channel->pending_sends.erase(ctx->msg_serial_id);
            }
            delete ctx;
          } else {
            auto *buf = (RoCEBuffer *)wc[i].wr_id;
            delete buf;
          }
        }
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
      uint32_t endpoint_len;
      co_await asio::async_read(*socket, asio::buffer(&endpoint_len, sizeof(uint32_t)),
                                asio::use_awaitable);
      std::string peer_endpoint_buf(endpoint_len, '\0');
      co_await asio::async_read(*socket, asio::buffer(peer_endpoint_buf.data(), endpoint_len),
                                asio::use_awaitable);
      nlohmann::json peer_endpoint_json = nlohmann::json::parse(peer_endpoint_buf);
      Endpoint peer_endpoint = Endpoint::from_json(peer_endpoint_json);
      auto channel = std::make_shared<RoCEChannel>();
      channel->endpoint = peer_endpoint;
      channel->qp = create_qp();
      RoCEChannelInfo my_info = get_local_info(channel->qp);
      uint32_t my_psn = my_info.psn;

      IBuffer my_info_buf(out_allocator_, 26);
      serialize_info(my_info, my_info_buf);
      co_await asio::async_write(*socket, asio::buffer(my_info_buf.data(), my_info_buf.size()),
                                 asio::use_awaitable);
      IBuffer peer_info_buf(out_allocator_, 26);
      co_await asio::async_read(*socket, asio::buffer(peer_info_buf.data(), peer_info_buf.size()),
                                asio::use_awaitable);
      RoCEChannelInfo peer_info = deserialize_info(peer_info_buf);
      modify_qp_to_rts(channel->qp, peer_info, my_psn);
      for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
        auto buf = std::make_unique<RoCEBuffer>(*ibv_allocator_, ROCE_BUFFER_SIZE);
        buf->resize(ROCE_BUFFER_SIZE);
        post_recv_buffer(channel->qp, buf.get());
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
