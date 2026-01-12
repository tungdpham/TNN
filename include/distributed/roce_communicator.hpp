/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "communicator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/fragmenter.hpp"
#include "distributed/packet.hpp"
#include "distributed/roce_buffer_pool.hpp"
#include "endpoint.hpp"
#include <any>
#include <cerrno>

#include <asio.hpp>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <infiniband/verbs.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tnn {

constexpr int ROCE_BUFFER_SIZE = 16 * 1024 * 1024; // 16MB
constexpr int ROCE_SQ_DEPTH = 128;
constexpr int ROCE_RQ_DEPTH = 128;

struct RoceConnectionInfo {
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
  union ibv_gid gid;
};

class RoceCommunicator : public Communicator {
private:
  struct RegisteredBuffer {
    std::vector<uint8_t> data;
    ibv_mr *mr = nullptr;
    std::any attached_context;

    RegisteredBuffer(size_t size, ibv_pd *pd) {
      data.resize(size);
      mr = ibv_reg_mr(pd, data.data(), size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (!mr) {
        throw std::runtime_error("Failed to register MR");
      }
    }

    ~RegisteredBuffer() {
      if (mr) {
        ibv_dereg_mr(mr);
      }
    }
  };

  struct Connection {
    ibv_qp *qp = nullptr;
    std::string peer_id;
    uint32_t psn = 0;
    std::vector<std::unique_ptr<RegisteredBuffer>> recv_buffers;
    Fragmenter fragmenter;

    ~Connection() {
      if (qp) {
        ibv_destroy_qp(qp);
        qp = nullptr;
      }
    }
  };

  std::string device_name_;
  int port_;
  int gid_index_;
  int ib_port_ = 1;

  ibv_context *context_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_cq *cq_ = nullptr;

  std::thread poll_thread_;
  std::atomic<bool> is_running_{false};

  std::unordered_map<std::string, std::shared_ptr<Connection>> connections_;
  std::unordered_map<uint32_t, std::shared_ptr<Connection>>
      qp_map_; // Map QPN to Connection* for polling
  std::mutex connections_mutex_;

  // Send buffer pool
  std::vector<std::unique_ptr<RegisteredBuffer>> send_buffers_;
  std::queue<RegisteredBuffer *> free_send_buffers_;
  std::mutex send_buffers_mutex_;
  std::condition_variable send_buffers_cv_;

  std::unique_ptr<RoceBufferPool> buffer_pool_;

  asio::io_context io_context_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread io_thread_;

public:
  explicit RoceCommunicator(const std::string &id, const Endpoint &endpoint)
      : Communicator(id), acceptor_(io_context_) {
    try {
      device_name_ = endpoint.get_parameter<std::string>("device_name");
      port_ = endpoint.get_parameter<int>("port");
      gid_index_ = endpoint.get_parameter<int>("gid_index");
    } catch (...) {
      // Fallback or rethrow?
      throw;
    }

    init_rdma();
    print_gid_table();
    buffer_pool_ = std::make_unique<RoceBufferPool>(pd_);
    init_send_buffers();

    is_running_ = true;
    poll_thread_ = std::thread(&RoceCommunicator::poll_cq, this);
  }

  ~RoceCommunicator() override { stop(); }

  void stop() {
    is_running_ = false;
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }

    io_context_.stop();
    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      connections_.clear();
      qp_map_.clear();
    }

    send_buffers_.clear();
    buffer_pool_.reset();

    if (cq_)
      ibv_destroy_cq(cq_);
    if (pd_)
      ibv_dealloc_pd(pd_);
    if (context_)
      ibv_close_device(context_);
  }

  void start_server() {
    asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port_);
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();

    accept_connections();
    io_thread_ = std::thread([this]() { io_context_.run(); });
  }

  void send_message(Message &&message) override {
    size_t msg_size = message.size();
    PooledRoceBuffer data_buffer = buffer_pool_->get_buffer(msg_size);
    size_t offset = 0;
    BinarySerializer::serialize(*data_buffer, offset, message);

    size_t max_payload = ROCE_BUFFER_SIZE - sizeof(PacketHeader);
    uint32_t num_packets = (data_buffer->size() + max_payload - 1) / max_payload;
    if (num_packets == 0)
      num_packets = 1;

    std::vector<PacketHeader> headers;
    std::shared_ptr<Connection> conn;
    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      auto it = connections_.find(message.header().recipient_id);
      if (it != connections_.end()) {
        conn = it->second;
        headers = conn->fragmenter.get_headers(*data_buffer, num_packets);
      }
    }

    if (!conn) {
      std::cerr << "Connection not found for recipient: " << message.header().recipient_id
                << std::endl;
      return;
    }

    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    if (ibv_query_qp(conn->qp, &attr, IBV_QP_STATE, &init_attr) == 0) {
      if (attr.qp_state != IBV_QPS_RTS) {
        std::cerr << "QP for " << message.header().recipient_id
                  << " not in RTS state (state: " << attr.qp_state << "), dropping message\\n";
        return;
      }
    }

    for (size_t i = 0; i < headers.size(); ++i) {
      RegisteredBuffer *send_buf = nullptr;
      {
        std::unique_lock<std::mutex> lock(send_buffers_mutex_);
        send_buffers_cv_.wait(lock, [this] { return !free_send_buffers_.empty(); });
        send_buf = free_send_buffers_.front();
        free_send_buffers_.pop();
      }

      send_buf->attached_context = data_buffer;

      // Copy header
      std::memcpy(send_buf->data.data(), &headers[i], sizeof(PacketHeader));

      struct ibv_sge sge[2];
      sge[0].addr = (uint64_t)send_buf->data.data();
      sge[0].length = sizeof(PacketHeader);
      sge[0].lkey = send_buf->mr->lkey;

      size_t copy_len = headers[i].length;
      if (copy_len > 0) {
        sge[1].addr = (uint64_t)(data_buffer->get() + headers[i].packet_offset);
        sge[1].length = copy_len;
        sge[1].lkey = data_buffer->get_lkey();
      }

      struct ibv_send_wr wr, *bad_wr = nullptr;
      std::memset(&wr, 0, sizeof(wr));
      wr.wr_id = (uint64_t)send_buf;
      wr.opcode = IBV_WR_SEND;
      wr.sg_list = sge;
      wr.num_sge = (copy_len > 0) ? 2 : 1;
      wr.send_flags = IBV_SEND_SIGNALED;

      if (ibv_post_send(conn->qp, &wr, &bad_wr)) {
        std::cerr << "Failed to post send to " << message.header().recipient_id << std::endl;
        send_buf->attached_context.reset();
        std::lock_guard<std::mutex> lock(send_buffers_mutex_);
        free_send_buffers_.push(send_buf);
        send_buffers_cv_.notify_one();
      }
    }
  }

  void flush_output_messages() override {
    std::lock_guard<std::mutex> lock(out_message_mutex_);
    while (!out_message_queue_.empty()) {
      auto message = std::move(out_message_queue_.front());
      send_message(std::move(message));
      out_message_queue_.pop();
    }
  }

protected:
  bool connect_to_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    try {
      std::string host = endpoint.get_parameter<std::string>("host");
      int tcp_port = endpoint.get_parameter<int>("port");

      asio::ip::tcp::socket socket(io_context_);
      asio::ip::tcp::resolver resolver(io_context_);
      asio::connect(socket, resolver.resolve(host, std::to_string(tcp_port)));

      // Handshake
      uint32_t id_len = id_.length();
      asio::write(socket, asio::buffer(&id_len, sizeof(id_len)));
      asio::write(socket, asio::buffer(id_));

      auto conn = std::make_shared<Connection>();
      conn->peer_id = peer_id;
      conn->qp = create_qp();

      RoceConnectionInfo my_info = get_local_info(conn->qp);
      std::vector<uint8_t> my_info_buf;
      serialize_info(my_info, my_info_buf);
      asio::write(socket, asio::buffer(my_info_buf));

      std::vector<uint8_t> peer_info_buf(26);
      asio::read(socket, asio::buffer(peer_info_buf));
      RoceConnectionInfo peer_info = deserialize_info(peer_info_buf);

      modify_qp_to_rts(conn->qp, peer_info, my_info.psn);

      // Initialize receive buffers for this connection
      for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
        auto buf = std::make_unique<RegisteredBuffer>(ROCE_BUFFER_SIZE, pd_);
        post_recv_buffer(conn->qp, buf.get());
        conn->recv_buffers.push_back(std::move(buf));
      }

      // Send ACK to indicate receive buffers are ready
      uint8_t ack = 1;
      asio::write(socket, asio::buffer(&ack, 1));

      // Wait for remote ACK
      uint8_t remote_ack = 0;
      asio::read(socket, asio::buffer(&remote_ack, 1));

      if (remote_ack != 1) {
        throw std::runtime_error("Invalid ACK from remote peer");
      }

      {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        qp_map_[conn->qp->qp_num] = conn;
        connections_[peer_id] = conn;
      }

      std::cout << "Successfully established outgoing connection to " << peer_id
                << " (QP: " << conn->qp->qp_num << ")\n";
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Connect error: " << e.what() << std::endl;
      return false;
    }
  }

  bool disconnect_from_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(peer_id);
    if (it != connections_.end()) {
      qp_map_.erase(it->second->qp->qp_num);
      connections_.erase(it);
      return true;
    }
    return false;
  }

private:
  void serialize_info(const RoceConnectionInfo &info, std::vector<uint8_t> &buf) {
    buf.reserve(26);
    // lid (16)
    buf.push_back((info.lid >> 8) & 0xFF);
    buf.push_back(info.lid & 0xFF);
    // qpn (32)
    buf.push_back((info.qpn >> 24) & 0xFF);
    buf.push_back((info.qpn >> 16) & 0xFF);
    buf.push_back((info.qpn >> 8) & 0xFF);
    buf.push_back(info.qpn & 0xFF);
    // psn (32)
    buf.push_back((info.psn >> 24) & 0xFF);
    buf.push_back((info.psn >> 16) & 0xFF);
    buf.push_back((info.psn >> 8) & 0xFF);
    buf.push_back(info.psn & 0xFF);
    // gid (16 bytes)
    const uint8_t *gid_ptr = info.gid.raw;
    for (int i = 0; i < 16; ++i)
      buf.push_back(gid_ptr[i]);
  }

  RoceConnectionInfo deserialize_info(const std::vector<uint8_t> &buf) {
    RoceConnectionInfo info;
    int idx = 0;
    info.lid = (buf[idx] << 8) | buf[idx + 1];
    idx += 2;
    info.qpn = (buf[idx] << 24) | (buf[idx + 1] << 16) | (buf[idx + 2] << 8) | buf[idx + 3];
    idx += 4;
    info.psn = (buf[idx] << 24) | (buf[idx + 1] << 16) | (buf[idx + 2] << 8) | buf[idx + 3];
    idx += 4;
    std::memcpy(info.gid.raw, &buf[idx], 16);
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
            if (gid.raw[b] != 0)
              empty = false;
          if (empty)
            continue;

          std::cout << "  GID Index " << i << ": ";
          auto old_flags = std::cout.flags();
          std::cout << std::hex;
          for (int b = 0; b < 16; ++b)
            std::cout << (int)gid.raw[b] << (b < 15 ? ":" : "");
          std::cout.flags(old_flags);
          std::cout << "\n";
        }
      }
    }
  }

  void init_rdma() {
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list)
      throw std::runtime_error("Failed to get IB devices list");

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
    if (!context_)
      throw std::runtime_error("Failed to open device");

    pd_ = ibv_alloc_pd(context_);
    if (!pd_)
      throw std::runtime_error("Failed to alloc PD");

    cq_ = ibv_create_cq(context_, ROCE_SQ_DEPTH + ROCE_RQ_DEPTH, nullptr, nullptr, 0);
    if (!cq_)
      throw std::runtime_error("Failed to create CQ");

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
            if (empty)
              continue;

            // check for IPv4 mapped address ::ffff:x.x.x.x
            // this usually indicates RoCE v2 with IPv4
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

  void init_send_buffers() {
    for (int i = 0; i < ROCE_SQ_DEPTH; ++i) {
      auto buf = std::make_unique<RegisteredBuffer>(ROCE_BUFFER_SIZE, pd_);
      free_send_buffers_.push(buf.get());
      send_buffers_.push_back(std::move(buf));
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
    if (!qp)
      throw std::runtime_error("Failed to create QP");
    return qp;
  }

  RoceConnectionInfo get_local_info(ibv_qp *qp) {
    RoceConnectionInfo info;
    info.qpn = qp->qp_num;
    info.psn = lrand48() & 0xffffff;

    struct ibv_port_attr attr;
    ibv_query_port(context_, ib_port_, &attr);
    info.lid = attr.lid;

    ibv_query_gid(context_, ib_port_, gid_index_, &info.gid);
    return info;
  }

  void modify_qp_to_rts(ibv_qp *qp, const RoceConnectionInfo &peer_info, uint32_t local_psn) {
    struct ibv_qp_attr attr;
    int flags;
    int ret;

    // check port attributes to determine MTU
    struct ibv_port_attr port_attr;
    if ((ret = ibv_query_port(context_, ib_port_, &port_attr)) != 0) {
      throw std::runtime_error("Failed to query port: " + std::string(std::strerror(ret)));
    }

    // INIT
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

    // RTR
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
      std::cerr << std::hex;
      for (int i = 0; i < 16; ++i)
        std::cerr << (int)peer_info.gid.raw[i] << ":";
      std::cerr.flags(old_flags);
      std::cerr << "\n";

      union ibv_gid local_gid;
      if (ibv_query_gid(context_, ib_port_, gid_index_, &local_gid) == 0) {
        std::cerr << "  Local GID (Index " << gid_index_ << "): ";
        std::cerr << std::hex;
        for (int i = 0; i < 16; ++i)
          std::cerr << (int)local_gid.raw[i] << ":";
        std::cerr.flags(old_flags);
        std::cerr << "\n";

        bool remote_ipv4 = true;
        for (int i = 0; i < 10; ++i)
          if (peer_info.gid.raw[i] != 0)
            remote_ipv4 = false;
        if (peer_info.gid.raw[10] != 0xff || peer_info.gid.raw[11] != 0xff)
          remote_ipv4 = false;

        bool local_ipv4 = true;
        for (int i = 0; i < 10; ++i)
          if (local_gid.raw[i] != 0)
            local_ipv4 = false;
        if (local_gid.raw[10] != 0xff || local_gid.raw[11] != 0xff)
          local_ipv4 = false;

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

    // RTS
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 20;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn; // My PSN
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
            IBV_QP_MAX_QP_RD_ATOMIC;
    if ((ret = ibv_modify_qp(qp, &attr, flags)) != 0) {
      std::cerr << "Failed to modify QP to RTS. ret=" << ret << " (" << std::strerror(ret) << ")\n";
      throw std::runtime_error("Failed to modify QP to RTS");
    }
  }

  void post_recv_buffer(ibv_qp *qp, RegisteredBuffer *buf) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)buf->data.data();
    sge.length = ROCE_BUFFER_SIZE;
    sge.lkey = buf->mr->lkey;

    struct ibv_recv_wr wr, *bad_wr = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)buf;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to post recv" << std::endl;
    }
  }

  void poll_cq() {
    struct ibv_wc wc[16];
    while (is_running_) {
      int n = ibv_poll_cq(cq_, 16, wc);
      if (n < 0) {
        std::cerr << "Poll CQ failed" << std::endl;
        break;
      }
      for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::cerr << "WC Error: " << ibv_wc_status_str(wc[i].status) << " for QP " << wc[i].qp_num
                    << std::endl;
          {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            auto it = qp_map_.find(wc[i].qp_num);
            if (it != qp_map_.end()) {
              std::string peer_id = it->second->peer_id;
              qp_map_.erase(it);
              connections_.erase(peer_id);
            }
          }
          continue;
        }

        if (wc[i].opcode & IBV_WC_RECV) {
          auto *buf = (RegisteredBuffer *)wc[i].wr_id;
          PacketHeader header;
          std::memcpy(&header, buf->data.data(), sizeof(PacketHeader));

          std::shared_ptr<Connection> conn;
          {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            auto it = qp_map_.find(wc[i].qp_num);
            if (it != qp_map_.end())
              conn = it->second;
          }

          if (conn) {
            if (conn->fragmenter.message_exists(header.msg_serial_id) ||
                header.packet_offset == 0) {
              conn->fragmenter.register_packet(header.msg_serial_id, header);
              auto dest_buf = conn->fragmenter.get_packet_buffer(header.msg_serial_id, header);

              size_t payload_len = header.length;
              std::memcpy(dest_buf->get() + header.packet_offset,
                          buf->data.data() + sizeof(PacketHeader), payload_len);

              if (conn->fragmenter.commit_packet(header.msg_serial_id, header)) {
                // Message complete
                MessageState state = conn->fragmenter.fetch_complete_message(header.msg_serial_id);
                Message msg;
                size_t offset = 0;
                BinarySerializer::deserialize(*state.buffer, offset, msg);
                enqueue_input_message(std::move(msg));
              }
            }
            // Repost recv buffer
            post_recv_buffer(conn->qp, buf);
          } else {
            std::cerr << "Received packet from unknown QP: " << wc[i].qp_num << std::endl;
          }

        } else if (wc[i].opcode == IBV_WC_SEND || wc[i].opcode == IBV_WC_RDMA_WRITE) {
          auto *buf = (RegisteredBuffer *)wc[i].wr_id;
          buf->attached_context.reset();
          std::lock_guard<std::mutex> lock(send_buffers_mutex_);
          free_send_buffers_.push(buf);
          send_buffers_cv_.notify_one();
        }
      }
    }
  }

  void accept_connections() {
    auto socket = std::make_shared<asio::ip::tcp::socket>(io_context_);
    acceptor_.async_accept(*socket, [this, socket](const asio::error_code &error) {
      if (!error) {
        handle_new_connection(socket);
      }
      if (is_running_) {
        accept_connections();
      }
    });
  }

  void handle_new_connection(std::shared_ptr<asio::ip::tcp::socket> socket) {
    std::cout << "Handling new connection: \n";

    auto id_len_buf = std::make_shared<uint32_t>();
    asio::async_read(
        *socket, asio::buffer(id_len_buf.get(), sizeof(uint32_t)),
        [this, socket, id_len_buf](const asio::error_code &ec, size_t) {
          if (ec)
            return;
          size_t id_len = *id_len_buf;
          auto peer_id_buf = std::make_shared<std::string>(id_len, '\0');
          asio::async_read(
              *socket, asio::buffer(&(*peer_id_buf)[0], id_len),
              [this, socket, peer_id_buf](const asio::error_code &ec, size_t) {
                if (ec)
                  return;
                std::string peer_id = *peer_id_buf;
                std::cout << "Incoming connection from: " << peer_id << std::endl;

                auto conn = std::make_shared<Connection>();
                conn->peer_id = peer_id;
                try {
                  conn->qp = create_qp();
                } catch (...) {
                  std::cerr << "Error while trying to create qp" << std::endl;
                  return;
                }

                RoceConnectionInfo my_info = get_local_info(conn->qp);
                uint32_t my_psn = my_info.psn;
                auto my_info_buf = std::make_shared<std::vector<uint8_t>>();
                serialize_info(my_info, *my_info_buf);

                asio::async_write(
                    *socket, asio::buffer(*my_info_buf),
                    [this, socket, conn, peer_id, my_psn](const asio::error_code &ec, size_t) {
                      if (ec) {
                        std::cerr << "Error during async_write: " << ec.message() << std::endl;
                        return;
                      }
                      auto peer_info_buf = std::make_shared<std::vector<uint8_t>>(26);
                      asio::async_read(
                          *socket, asio::buffer(*peer_info_buf),
                          [this, socket, conn, peer_id, peer_info_buf,
                           my_psn](const asio::error_code &ec, size_t) {
                            if (ec) {
                              std::cerr << "Error during async_read: " << ec.message() << std::endl;
                              return;
                            }
                            try {
                              RoceConnectionInfo peer_info = deserialize_info(*peer_info_buf);
                              modify_qp_to_rts(conn->qp, peer_info, my_psn);

                              for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
                                auto buf =
                                    std::make_unique<RegisteredBuffer>(ROCE_BUFFER_SIZE, pd_);
                                post_recv_buffer(conn->qp, buf.get());
                                conn->recv_buffers.push_back(std::move(buf));
                              }

                              // Wait for remote ACK that their receive buffers are ready
                              auto remote_ack_buf = std::make_shared<uint8_t>(0);
                              asio::async_read(
                                  *socket, asio::buffer(remote_ack_buf.get(), 1),
                                  [this, socket, conn, peer_id,
                                   remote_ack_buf](const asio::error_code &ec, size_t) {
                                    if (ec) {
                                      std::cerr << "Error reading ACK: " << ec.message()
                                                << std::endl;
                                      return;
                                    }

                                    // Send our ACK that we're ready
                                    auto ack_buf = std::make_shared<uint8_t>(1);
                                    asio::async_write(
                                        *socket, asio::buffer(ack_buf.get(), 1),
                                        [this, conn, peer_id, ack_buf](const asio::error_code &ec,
                                                                       size_t) {
                                          if (ec) {
                                            std::cerr << "Error sending ACK: " << ec.message()
                                                      << std::endl;
                                            return;
                                          }

                                          {
                                            std::lock_guard<std::mutex> lock(connections_mutex_);
                                            qp_map_[conn->qp->qp_num] = conn;
                                            connections_[peer_id] = conn;
                                          }

                                          std::cout << "Successfully established incoming "
                                                       "connection from "
                                                    << peer_id << " (QP: " << conn->qp->qp_num
                                                    << ")\n";
                                        });
                                  });

                            } catch (const std::exception &e) {
                              std::cerr << "Handshake error: " << e.what() << std::endl;
                            }
                          });
                    });
              });
        });
  }
};

} // namespace tnn
