/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "communicator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/buffer_pool.hpp"
#include "distributed/fragmenter.hpp"
#include "distributed/packet.hpp"
#include "endpoint.hpp"

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

constexpr int ROCE_BUFFER_SIZE = 1024 * 1024; // 1MB
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

  std::unordered_map<std::string, std::unique_ptr<Connection>> connections_;
  std::unordered_map<uint32_t, Connection *> qp_map_; // Map QPN to Connection* for polling
  std::mutex connections_mutex_;

  // Send buffer pool
  std::vector<std::unique_ptr<RegisteredBuffer>> send_buffers_;
  std::queue<RegisteredBuffer *> free_send_buffers_;
  std::mutex send_buffers_mutex_;
  std::condition_variable send_buffers_cv_;

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
      for (auto &pair : connections_) {
        if (pair.second->qp)
          ibv_destroy_qp(pair.second->qp);
      }
      connections_.clear();
      qp_map_.clear();
    }

    send_buffers_.clear();

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

  void send_message(const Message &message) override {
    size_t msg_size = message.size();
    PooledBuffer data_buffer = BufferPool::instance().get_buffer(msg_size);
    BinarySerializer::serialize(message, *data_buffer);

    size_t max_payload = ROCE_BUFFER_SIZE - sizeof(PacketHeader);
    uint32_t num_packets = (data_buffer->size() + max_payload - 1) / max_payload;
    if (num_packets == 0)
      num_packets = 1;

    std::vector<PacketHeader> headers;
    Connection *conn = nullptr;
    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      auto it = connections_.find(message.header().recipient_id);
      if (it != connections_.end()) {
        conn = it->second.get();
        headers = conn->fragmenter.get_headers(*data_buffer, num_packets);
      }
    }

    if (!conn) {
      std::cerr << "Connection not found for recipient: " << message.header().recipient_id
                << std::endl;
      return;
    }

    for (size_t i = 0; i < headers.size(); ++i) {
      RegisteredBuffer *send_buf = nullptr;
      {
        std::unique_lock<std::mutex> lock(send_buffers_mutex_);
        send_buffers_cv_.wait(lock, [this] { return !free_send_buffers_.empty(); });
        send_buf = free_send_buffers_.front();
        free_send_buffers_.pop();
      }

      // Copy header
      std::memcpy(send_buf->data.data(), &headers[i], sizeof(PacketHeader));
      // Copy data
      size_t copy_len = headers[i].length;
      std::memcpy(send_buf->data.data() + sizeof(PacketHeader),
                  data_buffer->get() + headers[i].packet_offset, copy_len);

      struct ibv_sge sge;
      sge.addr = (uint64_t)send_buf->data.data();
      sge.length = sizeof(PacketHeader) + copy_len;
      sge.lkey = send_buf->mr->lkey;

      struct ibv_send_wr wr, *bad_wr = nullptr;
      std::memset(&wr, 0, sizeof(wr));
      wr.wr_id = (uint64_t)send_buf;
      wr.opcode = IBV_WR_SEND;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.send_flags = IBV_SEND_SIGNALED;

      if (ibv_post_send(conn->qp, &wr, &bad_wr)) {
        std::cerr << "Failed to post send to " << message.header().recipient_id << std::endl;
        std::lock_guard<std::mutex> lock(send_buffers_mutex_);
        free_send_buffers_.push(send_buf);
        send_buffers_cv_.notify_one();
      }
    }
  }

  void flush_output_messages() override {
    std::lock_guard<std::mutex> lock(out_message_mutex_);
    while (!out_message_queue_.empty()) {
      send_message(out_message_queue_.front());
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

      auto conn = std::make_unique<Connection>();
      conn->peer_id = peer_id;
      conn->qp = create_qp();

      RoceConnectionInfo my_info = get_local_info(conn->qp);
      asio::write(socket, asio::buffer(&my_info, sizeof(my_info)));

      RoceConnectionInfo peer_info;
      asio::read(socket, asio::buffer(&peer_info, sizeof(peer_info)));

      modify_qp_to_rts(conn->qp, peer_info);

      // Initialize receive buffers for this connection
      for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
        auto buf = std::make_unique<RegisteredBuffer>(ROCE_BUFFER_SIZE, pd_);
        post_recv_buffer(conn->qp, buf.get());
        conn->recv_buffers.push_back(std::move(buf));
      }

      {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        qp_map_[conn->qp->qp_num] = conn.get();
        connections_[peer_id] = std::move(conn);
      }

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
      ibv_destroy_qp(it->second->qp);
      connections_.erase(it);
      return true;
    }
    return false;
  }

private:
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
    init_attr.cap.max_send_sge = 1;
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

  void modify_qp_to_rts(ibv_qp *qp, const RoceConnectionInfo &peer_info) {
    struct ibv_qp_attr attr;
    int flags;

    // INIT
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ib_port_;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp, &attr, flags))
      throw std::runtime_error("Failed to modify QP to INIT");

    // RTR
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
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
    attr.ah_attr.grh.hop_limit = 1;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp, &attr, flags))
      throw std::runtime_error("Failed to modify QP to RTR");

    // RTS
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0; // My PSN
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
            IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp, &attr, flags))
      throw std::runtime_error("Failed to modify QP to RTS");
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
          std::cerr << "WC Error: " << ibv_wc_status_str(wc[i].status) << std::endl;
          continue;
        }

        if (wc[i].opcode & IBV_WC_RECV) {
          auto *buf = (RegisteredBuffer *)wc[i].wr_id;
          PacketHeader header;
          std::memcpy(&header, buf->data.data(), sizeof(PacketHeader));

          Connection *conn = nullptr;
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
              auto &dest_buf = conn->fragmenter.get_packet_buffer(header.msg_serial_id, header);

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

        } else if (wc[i].opcode == IBV_WR_SEND) {
          auto *buf = (RegisteredBuffer *)wc[i].wr_id;
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
    std::thread([this, socket]() {
      try {
        uint32_t id_len;
        asio::read(*socket, asio::buffer(&id_len, sizeof(id_len)));
        std::string peer_id(id_len, '\0');
        asio::read(*socket, asio::buffer(&peer_id[0], id_len));

        auto conn = std::make_unique<Connection>();
        conn->peer_id = peer_id;
        conn->qp = create_qp();

        RoceConnectionInfo my_info = get_local_info(conn->qp);
        asio::write(*socket, asio::buffer(&my_info, sizeof(my_info)));

        RoceConnectionInfo peer_info;
        asio::read(*socket, asio::buffer(&peer_info, sizeof(peer_info)));

        modify_qp_to_rts(conn->qp, peer_info);

        // Initialize receive buffers for this connection
        for (int i = 0; i < ROCE_RQ_DEPTH; ++i) {
          auto buf = std::make_unique<RegisteredBuffer>(ROCE_BUFFER_SIZE, pd_);
          post_recv_buffer(conn->qp, buf.get());
          conn->recv_buffers.push_back(std::move(buf));
        }

        {
          std::lock_guard<std::mutex> lock(connections_mutex_);
          qp_map_[conn->qp->qp_num] = conn.get();
          connections_[peer_id] = std::move(conn);
        }

        register_recipient(peer_id, Endpoint::roce(device_name_, port_, gid_index_));

      } catch (const std::exception &e) {
        std::cerr << "Handshake error: " << e.what() << std::endl;
      }
    }).detach();
  }
};

} // namespace tnn
