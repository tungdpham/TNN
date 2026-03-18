#include <infiniband/verbs.h>

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

#include "device/ibv_allocator.hpp"
#include "distributed/channel.hpp"
#include "distributed/io.hpp"
#include "distributed/packet.hpp"
#include "distributed/roce_cq.hpp"
#include "distributed/roce_device.hpp"

namespace tnn {

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

template <typename Archiver>
void archive(Archiver &archiver, const RoCEChannelInfo &info) {
  archiver(info.lid);
  archiver(info.qpn);
  archiver(info.psn);
  archiver(make_blob(info.gid.raw, 16));
}

template <typename Archiver>
void archive(Archiver &archiver, RoCEChannelInfo &info) {
  archiver(info.lid);
  archiver(info.qpn);
  archiver(info.psn);
  archiver(make_blob(info.gid.raw, 16));
}

struct WriteContext {
  uint64_t msg_serial_id;
};

class RoCEChannel : public Channel {
public:
  RoCEChannel(RoCEDevice &device, RoCECQ &cq, IbvAllocator &allocator)
      : device_(&device),
        context_(device.get_context()),
        ib_port_(device.get_port()),
        ibv_allocator_(allocator) {
    struct ibv_qp_init_attr init_attr;
    std::memset(&init_attr, 0, sizeof(init_attr));
    init_attr.send_cq = cq.handle();
    init_attr.recv_cq = cq.handle();
    init_attr.cap.max_send_wr = ROCE_SQ_DEPTH;
    init_attr.cap.max_recv_wr = ROCE_RQ_DEPTH;
    init_attr.cap.max_send_sge = 2;
    init_attr.cap.max_recv_sge = 1;
    init_attr.qp_type = IBV_QPT_RC;
    qp = ibv_create_qp(device.get_pd(), &init_attr);
    if (!qp) throw std::runtime_error("Failed to create QP");
  }

  ~RoCEChannel() {
    for (auto buf : recv_buffers) delete buf;
    for (auto p : pending_sends) delete p.second;
    for (auto p : pending_receives) delete p.second;

    if (qp) {
      ibv_destroy_qp(qp);
      qp = nullptr;
    }
  }

  void close() override {
    {
      std::lock_guard<std::mutex> lock(mutex);
      is_closed = true;
    }
    inflight_cv.notify_all();
    std::lock_guard<std::mutex> lock(mutex);

    for (auto &p : pending_sends) delete p.second;
    pending_sends.clear();

    for (auto &p : pending_receives) delete p.second;
    pending_receives.clear();
  }

  void enqueue_send(uint64_t msg_serial_id, dptr *data_buffer) {
    {
      std::unique_lock<std::mutex> conn_lock(mutex);
      inflight_cv.wait(conn_lock, [&] { return is_closed || inflight_count < 16; });
      if (is_closed) {
        delete data_buffer;
        return;
      }
      inflight_count++;
      pending_sends[msg_serial_id] = data_buffer;
    }
    PacketHeader header(PacketType::MSG_PREPARE, 0, data_buffer->capacity(), 0, 1);
    header.msg_serial_id = msg_serial_id;

    dptr *send_buf = new dptr(ibv_allocator_.allocate(sizeof(PacketHeader)));
    Writer header_writer(*send_buf);
    header_writer(header);

    struct ibv_sge sge;
    sge.addr = (uint64_t)send_buf->get();
    sge.length = sizeof(PacketHeader);
    sge.lkey = ibv_allocator_.get_mr_info(*send_buf).lkey;

    struct ibv_send_wr wr, *bad_wr = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)send_buf | 2;
    wr.opcode = IBV_WR_SEND;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    if (ibv_post_send(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to send MSG_PREPARE\n";
      delete send_buf;
      {
        std::lock_guard<std::mutex> conn_lock(mutex);
        inflight_count--;
        pending_sends.erase(header.msg_serial_id & 0xFFFFFFFF);
      }
      return;
    }
  }

  void handle_recv_wc(ibv_wc *wc, dptr *buf, const std::function<void(dptr *)> &on_payload_ready) {
    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      process_rdma_imm(ntohl(wc->imm_data), on_payload_ready);
    } else {
      PacketHeader header;
      Reader reader(*buf);
      reader(header);

      if (header.type == PacketType::MSG_PREPARE) {
        process_msg_prepare(header);
      } else if (header.type == PacketType::MSG_READY_TO_WRITE) {
        process_msg_ready(header, reader);
      }
    }
    // Automatically re-post the receive buffer
    post_recv_buffer(buf);
  }

  // Entry point for Send/RDMA_WRITE Work Completions
  void handle_send_wc(ibv_wc *wc) {
    uint64_t tag = wc->wr_id & 0x3;
    uint64_t ptr = wc->wr_id & ~0x3ULL;

    if (tag == 1) {  // RDMA_WRITE completion
      WriteContext *ctx = (WriteContext *)ptr;
      {
        std::lock_guard<std::mutex> lock(mutex);
        auto send_it = pending_sends.find(ctx->msg_serial_id);
        if (send_it != pending_sends.end()) {
          delete send_it->second;
          pending_sends.erase(send_it);
        }
        inflight_count--;
        inflight_cv.notify_one();
      }
      delete ctx;
    } else if (tag == 2) {  // MSG_PREPARE or MSG_READY completion
      delete (dptr *)ptr;
    }
  }

  void post_recv_buffer(dptr *buf) {
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

  RoCEChannelInfo get_local_info() const {
    RoCEChannelInfo info;
    info.qpn = qp->qp_num;
    info.psn = lrand48() & 0xffffff;
    struct ibv_port_attr attr;
    ibv_query_port(context_, ib_port_, &attr);
    info.lid = attr.lid;
    ibv_query_gid(context_, ib_port_, device_->get_gid_index(), &info.gid);
    return info;
  }

  void transition_to_rts(const RoCEChannelInfo &peer_info, uint32_t local_psn) {
    int gid_index = device_->get_gid_index();
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
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if ((ret = ibv_modify_qp(qp, &attr, flags)) != 0) {
      std::cerr << "Failed to modify QP to RTR. ret=" << ret << " (" << std::strerror(ret) << ")\n";
      std::cerr << "  GID Index: " << gid_index << "\n";
      std::cerr << "  Remote QPN: " << peer_info.qpn << "\n";
      std::cerr << "  Remote LID: " << peer_info.lid << "\n";
      std::cerr << "  MTU (Active/Path): " << port_attr.active_mtu << "/" << attr.path_mtu << "\n";
      std::cerr << "  Remote GID: ";
      auto old_flags = std::cerr.flags();
      for (int i = 0; i < 16; ++i) std::cerr << (int)peer_info.gid.raw[i] << ":";
      std::cerr.flags(old_flags);
      std::cerr << "\n";

      union ibv_gid local_gid;
      if (ibv_query_gid(context_, ib_port_, gid_index, &local_gid) == 0) {
        std::cerr << "  Local GID (Index " << gid_index << "): ";
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

  ibv_qp *qp = nullptr;
  Endpoint endpoint;
  uint32_t psn = 0;
  std::vector<dptr *> recv_buffers;

  std::mutex mutex;
  std::unordered_map<uint64_t, dptr *> pending_sends;
  std::unordered_map<uint32_t, dptr *> pending_receives;
  int inflight_count = 0;
  bool is_closed = false;
  std::condition_variable inflight_cv;

private:
  RoCEDevice *device_ = nullptr;
  ibv_context *context_ = nullptr;
  int ib_port_ = 0;
  IbvAllocator &ibv_allocator_;

  void process_rdma_imm(uint32_t imm, const std::function<void(dptr *)> &on_payload_ready) {
    dptr *dest_buf = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = pending_receives.find(imm);
      if (it != pending_receives.end()) {
        dest_buf = it->second;
        pending_receives.erase(it);
      } else {
        std::cerr << "No pending receive found for imm ID: " << imm << std::endl;
      }
    }
    // Pass the completed payload buffer back up to the communicator for deserialization
    if (dest_buf) {
      on_payload_ready(dest_buf);
    }
  }

  void process_msg_prepare(const PacketHeader &header) {
    dptr *dest_buf = new dptr(ibv_allocator_.allocate(header.msg_length));
    {
      std::lock_guard<std::mutex> lock(mutex);
      pending_receives[header.msg_serial_id & 0xFFFFFFFF] = dest_buf;
    }

    auto *send_buf = new dptr(ibv_allocator_.allocate(sizeof(PacketHeader) + 12));
    PacketHeader ready_header(PacketType::MSG_READY_TO_WRITE, 12, 0, 0, 1);
    ready_header.msg_serial_id = header.msg_serial_id;

    Writer writer(*send_buf);
    writer(ready_header);
    writer((uint64_t)dest_buf->get(), ibv_allocator_.get_mr_info(*dest_buf).rkey);

    struct ibv_sge sge;
    sge.addr = (uint64_t)send_buf->get();
    sge.length = sizeof(PacketHeader) + 12;
    sge.lkey = ibv_allocator_.get_mr_info(*send_buf).lkey;

    struct ibv_send_wr wr, *bad_wr = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)send_buf | 2;
    wr.opcode = IBV_WR_SEND;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;

    if (ibv_post_send(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to send MSG_READY\n";
      delete send_buf;
      std::lock_guard<std::mutex> lock(mutex);
      pending_receives.erase(header.msg_serial_id & 0xFFFFFFFF);
    }
  }

  void process_msg_ready(const PacketHeader &header, Reader &reader) {
    uint64_t remote_addr;
    uint32_t rkey;
    reader(remote_addr, rkey);

    dptr *source_buf;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = pending_sends.find(header.msg_serial_id);
      if (it != pending_sends.end()) {
        source_buf = it->second;
      } else {
        std::cerr << "No pending send found for MSG_READY with serial ID " << header.msg_serial_id
                  << "\n";
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

      if (ibv_post_send(qp, &wr, &bad_wr)) {
        std::cerr << "Failed to post RDMA write\n";
        delete ctx;
        return;
      }
    }
  }
};
}  // namespace tnn