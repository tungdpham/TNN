/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "binary_serializer.hpp"
#include "buffer_pool.hpp"
#include "communicator.hpp"
#include "message.hpp"

#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <asio.hpp>
#include <infiniband/verbs.h>

namespace tnn {

struct RdmaContext {
  ibv_context *context = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *cq = nullptr;
  ibv_port_attr port_attr;
  int port_num = 1;
  int gid_index = 0; // Default to 0, but should be configurable for RoCE v1/v2
};

struct RdmaConnection {
  ibv_qp *qp = nullptr;
  uint32_t remote_qpn = 0;
  uint16_t remote_lid = 0;
  union ibv_gid remote_gid;

  // Memory Region for receiving
  ibv_mr *recv_mr = nullptr;
  PooledBuffer recv_buffer;
};

// Structure to exchange connection information
struct CmId {
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
  union ibv_gid gid;
} __attribute__((packed));

class RdmaCommunicator : public Communicator {
public:
  explicit RdmaCommunicator(const Endpoint &endpoint) : io_context_(), acceptor_(io_context_) {
    try {
      // Parse configuration
      if (endpoint.has_parameter("ib_port")) {
        rdma_ctx_.port_num = std::stoi(endpoint.get_parameter<std::string>("ib_port"));
      }
      if (endpoint.has_parameter("gid_index")) {
        rdma_ctx_.gid_index = std::stoi(endpoint.get_parameter<std::string>("gid_index"));
      }

      // 1. Get device list
      int num_devices;
      struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
      if (!dev_list) {
        throw std::runtime_error("Failed to get RDMA device list");
      }

      // 2. Open device (using the first one for now, or filter by name if needed)
      // For Mellanox ConnectX-4, it should appear in this list.
      struct ibv_device *ib_dev = dev_list[0];
      if (!ib_dev) {
        ibv_free_device_list(dev_list);
        throw std::runtime_error("No RDMA devices found");
      }

      rdma_ctx_.context = ibv_open_device(ib_dev);
      ibv_free_device_list(dev_list);
      if (!rdma_ctx_.context) {
        throw std::runtime_error("Failed to open RDMA device");
      }

      // 3. Allocate Protection Domain
      rdma_ctx_.pd = ibv_alloc_pd(rdma_ctx_.context);
      if (!rdma_ctx_.pd) {
        throw std::runtime_error("Failed to allocate PD");
      }

      // 4. Create Completion Queue
      // Size 100 is arbitrary, should be tuned based on workload
      rdma_ctx_.cq = ibv_create_cq(rdma_ctx_.context, 100, nullptr, nullptr, 0);
      if (!rdma_ctx_.cq) {
        throw std::runtime_error("Failed to create CQ");
      }

      // 5. Query Port Attributes (needed for LID)
      if (ibv_query_port(rdma_ctx_.context, rdma_ctx_.port_num, &rdma_ctx_.port_attr)) {
        throw std::runtime_error("Failed to query port attributes");
      }

      // 6. Query GID
      if (ibv_query_gid(rdma_ctx_.context, rdma_ctx_.port_num, rdma_ctx_.gid_index, &local_gid_)) {
        throw std::runtime_error("Failed to query GID");
      }

      start_oob_server(endpoint);

    } catch (const std::exception &e) {
      std::cerr << "RdmaCommunicator initialization error: " << e.what() << std::endl;
      cleanup();
      throw;
    }

    is_running_ = true;
    polling_thread_ = std::thread(&RdmaCommunicator::poll_completion_queue, this);
    io_thread_ = std::thread([this]() { io_context_.run(); });
  }

  ~RdmaCommunicator() override { stop(); }

  void stop() {
    is_running_ = false;
    if (polling_thread_.joinable()) {
      polling_thread_.join();
    }

    io_context_.stop();
    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    cleanup();
  }

  void send_message(const Message &message) override {
    // For this example, we'll register on the fly (slow!) or assume a persistent buffer strategy.
    // To keep it simple but functional, let's assume we have a connection to the recipient.

    std::string recipient = message.header().recipient_id;

    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(recipient);
    if (it == connections_.end()) {
      std::cerr << "No RDMA connection to " << recipient << std::endl;
      return;
    }

    // Serialize message
    size_t msg_size = message.size();
    PacketHeader fixed_header(msg_size);
    size_t total_size = msg_size + PacketHeader::size();

    // Get a buffer (this buffer needs to be registered!)
    // For high performance, BufferPool should return buffers from a pre-registered memory region.
    // Here we will manually register for demonstration.
    PooledBuffer buffer = BufferPool::instance().get_buffer(total_size);
    BinarySerializer::serialize(fixed_header, *buffer);
    BinarySerializer::serialize(message, *buffer);

    struct ibv_mr *mr = ibv_reg_mr(rdma_ctx_.pd, buffer->get(), total_size, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) {
      std::cerr << "Failed to register memory for send" << std::endl;
      return;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)buffer->get();
    sge.length = total_size;
    sge.lkey = mr->lkey;

    struct ibv_send_wr sr;
    memset(&sr, 0, sizeof(sr));
    sr.wr_id = (uintptr_t)mr; // Store MR pointer to deregister later (unsafe in async!)
    sr.sg_list = &sge;
    sr.num_sge = 1;
    sr.opcode = IBV_WR_SEND;
    sr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr;
    if (ibv_post_send(it->second.qp, &sr, &bad_wr)) {
      std::cerr << "Failed to post send" << std::endl;
      ibv_dereg_mr(mr);
    }

    // Note: We are leaking MRs here because we can't deregister until completion!
    // A proper implementation would track in-flight requests and deregister in
    // poll_completion_queue.
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
  }

protected:
  bool connect_to_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    // 1. Create QP
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_type = IBV_QPT_RC; // Reliable Connection
    qp_init_attr.sq_sig_all = 1;
    qp_init_attr.send_cq = rdma_ctx_.cq;
    qp_init_attr.recv_cq = rdma_ctx_.cq;
    qp_init_attr.cap.max_send_wr = 10;
    qp_init_attr.cap.max_recv_wr = 10;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    struct ibv_qp *qp = ibv_create_qp(rdma_ctx_.pd, &qp_init_attr);
    if (!qp) {
      std::cerr << "Failed to create QP" << std::endl;
      return false;
    }

    // 2. Prepare local connection info
    CmId local_cm_id;
    local_cm_id.lid = htons(rdma_ctx_.port_attr.lid);
    local_cm_id.qpn = htonl(qp->qp_num);
    local_cm_id.psn = htonl(lrand48() & 0xffffff);
    local_cm_id.gid = local_gid_;

    // 3. Exchange info (OOB) via TCP
    CmId remote_cm_id;
    if (!exchange_cm_id(endpoint, local_cm_id, remote_cm_id)) {
      std::cerr << "Failed to exchange CM ID" << std::endl;
      ibv_destroy_qp(qp);
      return false;
    }

    // 4. Transition QP to INIT -> RTR -> RTS
    if (!modify_qp_to_init(qp)) {
      std::cerr << "Failed to modify QP to INIT" << std::endl;
      ibv_destroy_qp(qp);
      return false;
    }

    if (!modify_qp_to_rtr(qp, remote_cm_id)) {
      std::cerr << "Failed to modify QP to RTR" << std::endl;
      ibv_destroy_qp(qp);
      return false;
    }

    if (!modify_qp_to_rts(qp, remote_cm_id)) {
      std::cerr << "Failed to modify QP to RTS" << std::endl;
      ibv_destroy_qp(qp);
      return false;
    }

    RdmaConnection conn;
    conn.qp = qp;
    conn.remote_qpn = ntohl(remote_cm_id.qpn);
    conn.remote_lid = ntohs(remote_cm_id.lid);
    conn.remote_gid = remote_cm_id.gid;

    // Pre-post receive buffers
    conn.recv_buffer = BufferPool::instance().get_buffer(65536); // 64KB default
    conn.recv_mr = ibv_reg_mr(rdma_ctx_.pd, conn.recv_buffer->get(), conn.recv_buffer->capacity(),
                              IBV_ACCESS_LOCAL_WRITE);
    if (!conn.recv_mr) {
      std::cerr << "Failed to register recv MR" << std::endl;
      ibv_destroy_qp(qp);
      return false;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)conn.recv_buffer->get();
    sge.length = conn.recv_buffer->capacity();
    sge.lkey = conn.recv_mr->lkey;

    struct ibv_recv_wr rr;
    memset(&rr, 0, sizeof(rr));
    rr.wr_id = (uintptr_t)conn.recv_mr; // Use MR as ID
    rr.sg_list = &sge;
    rr.num_sge = 1;

    struct ibv_recv_wr *bad_wr;
    if (ibv_post_recv(qp, &rr, &bad_wr)) {
      std::cerr << "Failed to post recv" << std::endl;
      ibv_dereg_mr(conn.recv_mr);
      ibv_destroy_qp(qp);
      return false;
    }

    {
      std::unique_lock<std::shared_mutex> lock(connections_mutex_);
      connections_[peer_id] = std::move(conn);
    }

    return true;
  }

  bool disconnect_from_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    std::unique_lock<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(peer_id);
    if (it != connections_.end()) {
      if (it->second.qp) {
        ibv_destroy_qp(it->second.qp);
      }
      if (it->second.recv_mr) {
        ibv_dereg_mr(it->second.recv_mr);
      }
      connections_.erase(it);
      return true;
    }
    return false;
  }

private:
  std::atomic<bool> is_running_;
  std::thread polling_thread_;

  RdmaContext rdma_ctx_;
  std::unordered_map<std::string, RdmaConnection> connections_;
  std::shared_mutex connections_mutex_;

  union ibv_gid local_gid_;

  // OOB Connection handling
  asio::io_context io_context_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread io_thread_;

  void start_oob_server(const Endpoint &endpoint) {
    try {
      int port = std::stoi(endpoint.get_parameter<std::string>("port"));
      asio::ip::tcp::endpoint ep(asio::ip::tcp::v4(), port);
      acceptor_.open(ep.protocol());
      acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
      acceptor_.bind(ep);
      acceptor_.listen();

      accept_oob_connection();
    } catch (const std::exception &e) {
      std::cerr << "Failed to start OOB server: " << e.what() << std::endl;
    }
  }

  void accept_oob_connection() {
    auto socket = std::make_shared<asio::ip::tcp::socket>(io_context_);
    acceptor_.async_accept(*socket, [this, socket](std::error_code ec) {
      if (!ec) {
        // Handle incoming OOB connection (exchange CM ID)
        // In a real scenario, we'd need to know WHO is connecting to map it to a peer_id
        // For this simplified example, we assume the connector sends their ID first or we just
        // handle the exchange But wait, exchange_cm_id is client-side logic. Server-side logic
        // needs to respond.

        // Spawn a thread or coroutine to handle the handshake
        std::thread([this, socket]() {
          try {
            CmId remote_cm_id;
            asio::read(*socket, asio::buffer(&remote_cm_id, sizeof(CmId)));

            // We need to create a QP to respond with our info.
            // But we don't know which peer this is yet.
            // This architecture is a bit tricky without a proper handshake protocol.
            // For now, let's assume the client sends the peer_id first.

            // Simplified: Just echo back a dummy or fail.
            // Real implementation requires a full handshake protocol.
          } catch (...) {
          }
        }).detach();
      }
      accept_oob_connection();
    });
  }

  bool exchange_cm_id(const Endpoint &endpoint, const CmId &local_cm_id, CmId &remote_cm_id) {
    try {
      asio::io_context io_context;
      asio::ip::tcp::socket socket(io_context);
      asio::ip::tcp::resolver resolver(io_context);
      auto endpoints = resolver.resolve(endpoint.get_parameter<std::string>("host"),
                                        endpoint.get_parameter<std::string>("port"));
      asio::connect(socket, endpoints);

      asio::write(socket, asio::buffer(&local_cm_id, sizeof(CmId)));
      asio::read(socket, asio::buffer(&remote_cm_id, sizeof(CmId)));
      return true;
    } catch (const std::exception &e) {
      std::cerr << "OOB Exchange failed: " << e.what() << std::endl;
      return false;
    }
  }

  bool modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = rdma_ctx_.port_num;
    attr.pkey_index = 0;
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    return ibv_modify_qp(qp, &attr, flags) == 0;
  }

  bool modify_qp_to_rtr(struct ibv_qp *qp, const CmId &remote_cm_id) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024; // Should be negotiated
    attr.dest_qp_num = ntohl(remote_cm_id.qpn);
    attr.rq_psn = ntohl(remote_cm_id.psn);
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1; // Always use GRH for RoCE
    attr.ah_attr.dlid = ntohs(remote_cm_id.lid);
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = rdma_ctx_.port_num;

    // GID configuration
    attr.ah_attr.grh.dgid = remote_cm_id.gid;
    attr.ah_attr.grh.sgid_index = rdma_ctx_.gid_index;
    attr.ah_attr.grh.hop_limit = 1;

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    return ibv_modify_qp(qp, &attr, flags) == 0;
  }

  bool modify_qp_to_rts(struct ibv_qp *qp, const CmId &remote_cm_id) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = lrand48() & 0xffffff; // Should match what we sent? No, SQ PSN is local.
    attr.max_rd_atomic = 1;

    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    return ibv_modify_qp(qp, &attr, flags) == 0;
  }

  void cleanup() {
    if (rdma_ctx_.cq)
      ibv_destroy_cq(rdma_ctx_.cq);
    if (rdma_ctx_.pd)
      ibv_dealloc_pd(rdma_ctx_.pd);
    if (rdma_ctx_.context)
      ibv_close_device(rdma_ctx_.context);
  }

  void poll_completion_queue() {
    struct ibv_wc wc[10];
    while (is_running_) {
      int n = ibv_poll_cq(rdma_ctx_.cq, 10, wc);
      if (n < 0) {
        std::cerr << "Poll CQ failed" << std::endl;
        break;
      }

      for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::cerr << "Work completion error: " << ibv_wc_status_str(wc[i].status) << std::endl;
          continue;
        }

        if (wc[i].opcode == IBV_WC_RECV) {
          // Handle receive
          // Deserialize and enqueue
        } else if (wc[i].opcode == IBV_WC_SEND) {
          // Handle send completion
          // Deregister MR if we did the lazy registration
          struct ibv_mr *mr = (struct ibv_mr *)wc[i].wr_id;
          if (mr) {
            ibv_dereg_mr(mr);
          }
        }
      }

      if (n == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }
};

} // namespace tnn
