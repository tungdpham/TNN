#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static constexpr int kCtrlPort = 18515;

static void die(const std::string &msg) {
  std::cerr << "FATAL: " << msg << "\n";
  std::exit(1);
}

static std::string link_layer_to_str(uint8_t ll) {
  if (ll == IBV_LINK_LAYER_ETHERNET)
    return "Ethernet (RoCE)";
  if (ll == IBV_LINK_LAYER_INFINIBAND)
    return "InfiniBand";
  return "Unspecified/Other";
}

static const char *mtu_to_str(ibv_mtu m) {
  switch (m) {
  case IBV_MTU_256:
    return "256";
  case IBV_MTU_512:
    return "512";
  case IBV_MTU_1024:
    return "1024";
  case IBV_MTU_2048:
    return "2048";
  case IBV_MTU_4096:
    return "4096";
  default:
    return "unknown";
  }
}

static uint64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static void write_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = static_cast<const uint8_t *>(buf);
  size_t off = 0;
  while (off < len) {
    ssize_t n = ::send(fd, p + off, len - off, 0);
    if (n <= 0)
      die("TCP send failed");
    off += static_cast<size_t>(n);
  }
}

static void read_all(int fd, void *buf, size_t len) {
  uint8_t *p = static_cast<uint8_t *>(buf);
  size_t off = 0;
  while (off < len) {
    ssize_t n = ::recv(fd, p + off, len - off, MSG_WAITALL);
    if (n <= 0)
      die("TCP recv failed");
    off += static_cast<size_t>(n);
  }
}

static int tcp_listen(const std::string &bind_ip) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    die("socket() failed");
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kCtrlPort);
  if (::inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1)
    die("inet_pton bind_ip failed");
  if (::bind(fd, (sockaddr *)&addr, sizeof(addr)) != 0)
    die("bind() failed");
  if (::listen(fd, 1) != 0)
    die("listen() failed");
  return fd;
}

static int tcp_accept(int listen_fd) {
  sockaddr_in peer{};
  socklen_t sl = sizeof(peer);
  int cfd = ::accept(listen_fd, (sockaddr *)&peer, &sl);
  if (cfd < 0)
    die("accept() failed");
  return cfd;
}

static int tcp_connect(const std::string &server_ip) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    die("socket() failed");

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kCtrlPort);
  if (::inet_pton(AF_INET, server_ip.c_str(), &addr.sin_addr) != 1)
    die("inet_pton server_ip failed");
  if (::connect(fd, (sockaddr *)&addr, sizeof(addr)) != 0)
    die("connect() failed");
  return fd;
}

static void tcp_barrier(int sock, uint32_t cookie) {
  uint32_t x = htonl(cookie);
  write_all(sock, &x, sizeof(x));
  read_all(sock, &x, sizeof(x));
}

static uint32_t rand_psn() {
  return static_cast<uint32_t>((::getpid() ^ (now_ns() & 0xFFFFFFu)) & 0xFFFFFFu);
}

static uint64_t htonll_u64(uint64_t x) {
  uint32_t hi = htonl((uint32_t)(x >> 32));
  uint32_t lo = htonl((uint32_t)(x & 0xffffffffu));
  return (uint64_t)lo << 32 | hi;
}
static uint64_t ntohll_u64(uint64_t x) {
  uint32_t hi = ntohl((uint32_t)(x & 0xffffffffu));
  uint32_t lo = ntohl((uint32_t)(x >> 32));
  return (uint64_t)hi << 32 | lo;
}

struct CtrlMsg {
  uint32_t qpn;
  uint32_t psn;
  uint32_t active_mtu; // ibv_mtu enum
  uint32_t rkey;       // for RDMA WRITE (server's MR rkey)
  uint64_t raddr;      // for RDMA WRITE (server's buffer addr)
  uint8_t gid[16];
};

struct RdmaCtx {
  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;

  ibv_mr *mr_send = nullptr;
  ibv_mr *mr_recv = nullptr;

  void *buf_send = nullptr;
  void *buf_recv = nullptr;
  size_t buf_size = 0;

  uint8_t port_num = 1;
  int gid_index = -1;
  ibv_mtu active_mtu = IBV_MTU_256;

  ibv_gid local_gid{};
  uint32_t local_qpn = 0;
  uint32_t local_psn = 0;

  int qp_wr_limit = 0;
};

static ibv_device *pick_device(const std::string &dev_name) {
  int num = 0;
  ibv_device **list = ibv_get_device_list(&num);
  if (!list || num == 0)
    die("No RDMA devices found");
  ibv_device *chosen = nullptr;

  if (!dev_name.empty()) {
    for (int i = 0; i < num; i++) {
      if (dev_name == ibv_get_device_name(list[i])) {
        chosen = list[i];
        break;
      }
    }
    if (!chosen) {
      std::cerr << "Available devices:\n";
      for (int i = 0; i < num; i++)
        std::cerr << "  - " << ibv_get_device_name(list[i]) << "\n";
      die("Requested device not found: " + dev_name);
    }
  } else
    chosen = list[0];

  ibv_free_device_list(list);
  return chosen;
}

static void check_port_roce(ibv_context *ctx, uint8_t port) {
  ibv_port_attr pattr{};
  if (ibv_query_port(ctx, port, &pattr) != 0)
    die("ibv_query_port failed");

  std::cout << "Port " << int(port) << " state=" << pattr.state << " (4=ACTIVE)"
            << ", link_layer=" << link_layer_to_str(pattr.link_layer)
            << ", max_mtu=" << mtu_to_str((ibv_mtu)pattr.max_mtu)
            << ", active_mtu=" << mtu_to_str((ibv_mtu)pattr.active_mtu) << "\n";

  if (pattr.state != IBV_PORT_ACTIVE)
    die("Port is not ACTIVE");
  if (pattr.link_layer != IBV_LINK_LAYER_ETHERNET)
    die("Port link_layer is not Ethernet => not RoCE");
}

static std::string gid_to_hex(const ibv_gid &g) {
  char out[64]{};
  for (int i = 0; i < 16; i++)
    std::snprintf(out + i * 2, 3, "%02x", g.raw[i]);
  return std::string(out);
}

static int pick_gid_for_ipv4(ibv_context *ctx, uint8_t port, const std::string &ip4) {
  ibv_port_attr pattr{};
  if (ibv_query_port(ctx, port, &pattr) != 0)
    die("ibv_query_port failed in pick_gid_for_ipv4");

  in_addr ip{};
  if (inet_pton(AF_INET, ip4.c_str(), &ip) != 1)
    die("inet_pton failed for --local");

  uint8_t want[16]{};
  want[10] = 0xff;
  want[11] = 0xff;
  std::memcpy(&want[12], &ip.s_addr, 4);

  for (int i = 0; i < pattr.gid_tbl_len; i++) {
    ibv_gid g{};
    if (ibv_query_gid(ctx, port, i, &g) != 0)
      continue;
    if (std::memcmp(g.raw, want, 16) == 0)
      return i;
  }
  return -1;
}

static void alloc_buffers(RdmaCtx &r, size_t size, bool allow_remote_write) {
  r.buf_size = size;
  auto round4096 = [](size_t x) { return ((x + 4095) / 4096) * 4096; };
  size_t alloc_sz = round4096(size);

  r.buf_send = ::aligned_alloc(4096, alloc_sz);
  r.buf_recv = ::aligned_alloc(4096, alloc_sz);
  if (!r.buf_send || !r.buf_recv)
    die("aligned_alloc failed");

  std::memset(r.buf_send, 0xAB, size);
  std::memset(r.buf_recv, 0x00, size);

  int access = IBV_ACCESS_LOCAL_WRITE;
  if (allow_remote_write)
    access |= IBV_ACCESS_REMOTE_WRITE;

  r.mr_send = ibv_reg_mr(r.pd, r.buf_send, r.buf_size, IBV_ACCESS_LOCAL_WRITE);
  r.mr_recv = ibv_reg_mr(r.pd, r.buf_recv, r.buf_size, access);
  if (!r.mr_send || !r.mr_recv)
    die("ibv_reg_mr failed");
}

static void post_recv(RdmaCtx &r, uint32_t bytes, uint64_t wr_id) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(r.buf_recv);
  sge.length = bytes;
  sge.lkey = r.mr_recv->lkey;

  ibv_recv_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  ibv_recv_wr *bad = nullptr;
  if (ibv_post_recv(r.qp, &wr, &bad) != 0)
    die("ibv_post_recv failed");
}

static void post_send(RdmaCtx &r, uint32_t bytes, uint64_t wr_id) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(r.buf_send);
  sge.length = bytes;
  sge.lkey = r.mr_send->lkey;

  ibv_send_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  ibv_send_wr *bad = nullptr;
  if (ibv_post_send(r.qp, &wr, &bad) != 0)
    die("ibv_post_send failed");
}

static void post_write(RdmaCtx &r, uint32_t bytes, uint64_t remote_addr, uint32_t rkey,
                       uint64_t wr_id) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(r.buf_send);
  sge.length = bytes;
  sge.lkey = r.mr_send->lkey;

  ibv_send_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  ibv_send_wr *bad = nullptr;
  if (ibv_post_send(r.qp, &wr, &bad) != 0)
    die("ibv_post_send (RDMA_WRITE) failed");
}

static ibv_wc poll_cq_one(ibv_cq *cq) {
  ibv_wc wc{};
  while (true) {
    int n = ibv_poll_cq(cq, 1, &wc);
    if (n < 0)
      die("ibv_poll_cq failed");
    if (n == 0)
      continue;
    if (wc.status != IBV_WC_SUCCESS) {
      die("WC failed: status=" + std::to_string(wc.status) +
          " vendor_err=" + std::to_string(wc.vendor_err));
    }
    return wc;
  }
}

static void modify_qp_init(ibv_qp *qp, uint8_t port) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = port;
  attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE; // allow write from peer (safe even if peer doesn't)
  int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(qp, &attr, mask) != 0)
    die("ibv_modify_qp INIT failed");
}

static void modify_qp_rtr(ibv_qp *qp, uint32_t remote_qpn, uint32_t remote_psn,
                          const ibv_gid &remote_gid, uint8_t port, int gid_index,
                          ibv_mtu path_mtu) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = path_mtu;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = remote_psn;

  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.dgid = remote_gid;
  attr.ah_attr.grh.sgid_index = gid_index;
  attr.ah_attr.grh.hop_limit = 64;
  attr.ah_attr.port_num = port;

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
             IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  if (ibv_modify_qp(qp, &attr, mask) != 0)
    die("ibv_modify_qp RTR failed");
}

static void modify_qp_rts(ibv_qp *qp, uint32_t local_psn) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 18;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_psn;
  attr.max_rd_atomic = 1;

  int mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
             IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(qp, &attr, mask) != 0)
    die("ibv_modify_qp RTS failed");
}

static RdmaCtx rdma_setup(const std::string &dev_name, uint8_t port, int gid_index_or_auto,
                          const std::string &local_ip_for_gid, size_t buf_size,
                          bool allow_remote_write) {
  RdmaCtx r;
  r.port_num = port;
  r.gid_index = gid_index_or_auto;

  ibv_device *dev = pick_device(dev_name);
  r.ctx = ibv_open_device(dev);
  if (!r.ctx)
    die("ibv_open_device failed");

  ibv_device_attr dattr{};
  if (ibv_query_device(r.ctx, &dattr) != 0)
    die("ibv_query_device failed");
  std::cout << "Device: " << ibv_get_device_name(dev) << " | max_qp=" << dattr.max_qp
            << " | max_cq=" << dattr.max_cq << " | max_mr=" << dattr.max_mr
            << " | max_qp_wr=" << dattr.max_qp_wr << "\n";

  check_port_roce(r.ctx, port);

  ibv_port_attr pattr{};
  if (ibv_query_port(r.ctx, port, &pattr) != 0)
    die("ibv_query_port failed");
  r.active_mtu = (ibv_mtu)pattr.active_mtu;

  if (r.gid_index < 0) {
    if (local_ip_for_gid.empty())
      die("Auto GID needs --local <ipv4>");
    int idx = pick_gid_for_ipv4(r.ctx, port, local_ip_for_gid);
    if (idx < 0)
      die("Could not find GID matching ::ffff:" + local_ip_for_gid);
    r.gid_index = idx;
  }

  if (ibv_query_gid(r.ctx, port, r.gid_index, &r.local_gid) != 0)
    die("ibv_query_gid failed");

  r.pd = ibv_alloc_pd(r.ctx);
  if (!r.pd)
    die("ibv_alloc_pd failed");

  r.cq = ibv_create_cq(r.ctx, 1 << 15, nullptr, nullptr, 0);
  if (!r.cq)
    die("ibv_create_cq failed");

  int max_wr = (int)dattr.max_qp_wr;
  int want_wr = 8192;
  int wr = std::max(256, std::min(want_wr, max_wr));
  r.qp_wr_limit = wr;

  ibv_qp_init_attr qinit{};
  qinit.send_cq = r.cq;
  qinit.recv_cq = r.cq;
  qinit.qp_type = IBV_QPT_RC;
  qinit.cap.max_send_wr = wr;
  qinit.cap.max_recv_wr = wr;
  qinit.cap.max_send_sge = 1;
  qinit.cap.max_recv_sge = 1;

  r.qp = ibv_create_qp(r.pd, &qinit);
  if (!r.qp)
    die("ibv_create_qp failed");

  modify_qp_init(r.qp, port);
  alloc_buffers(r, buf_size, allow_remote_write);

  r.local_qpn = r.qp->qp_num;
  r.local_psn = rand_psn();

  std::cout << "Local QPN=" << r.local_qpn << " PSN=" << r.local_psn
            << " active_mtu=" << mtu_to_str(r.active_mtu) << " GID[" << r.gid_index
            << "]=" << gid_to_hex(r.local_gid) << " qp_wr_limit=" << r.qp_wr_limit << "\n";
  return r;
}

static void exchange_ctrl(int sock, const RdmaCtx &r, CtrlMsg &peer, bool i_am_server) {
  CtrlMsg me{};
  me.qpn = htonl(r.local_qpn);
  me.psn = htonl(r.local_psn);
  me.active_mtu = htonl((uint32_t)r.active_mtu);
  std::memcpy(me.gid, r.local_gid.raw, 16);

  // For RDMA WRITE: server publishes its recv buffer addr + rkey
  if (i_am_server) {
    me.rkey = htonl(r.mr_recv->rkey);
    me.raddr = htonll_u64((uint64_t)(uintptr_t)r.buf_recv);
  } else {
    me.rkey = htonl(0);
    me.raddr = htonll_u64(0);
  }

  write_all(sock, &me, sizeof(me));
  read_all(sock, &peer, sizeof(peer));

  peer.qpn = ntohl(peer.qpn);
  peer.psn = ntohl(peer.psn);
  peer.active_mtu = ntohl(peer.active_mtu);
  peer.rkey = ntohl(peer.rkey);
  peer.raddr = ntohll_u64(peer.raddr);
}

static void connect_qp(RdmaCtx &r, const CtrlMsg &peer) {
  ibv_gid remote_gid{};
  std::memcpy(remote_gid.raw, peer.gid, 16);

  ibv_mtu remote_mtu = (ibv_mtu)peer.active_mtu;
  ibv_mtu path_mtu = (r.active_mtu < remote_mtu) ? r.active_mtu : remote_mtu;

  std::cout << "Remote active_mtu=" << mtu_to_str(remote_mtu)
            << " => path_mtu=" << mtu_to_str(path_mtu) << "\n";

  modify_qp_rtr(r.qp, peer.qpn, peer.psn, remote_gid, r.port_num, r.gid_index, path_mtu);
  modify_qp_rts(r.qp, r.local_psn);
}

// Latency: SEND/RECV ping-pong 64B
static void latency_test(RdmaCtx &r, bool is_server, int iters) {
  const uint32_t bytes = 64;
  post_recv(r, bytes, 1);

  // warmup
  for (int i = 0; i < 200; i++) {
    if (!is_server) {
      post_send(r, bytes, 2);
      while (poll_cq_one(r.cq).opcode != IBV_WC_SEND) {
      }
      while (poll_cq_one(r.cq).opcode != IBV_WC_RECV) {
      }
      post_recv(r, bytes, 1);
    } else {
      while (poll_cq_one(r.cq).opcode != IBV_WC_RECV) {
      }
      post_recv(r, bytes, 1);
      post_send(r, bytes, 2);
      while (poll_cq_one(r.cq).opcode != IBV_WC_SEND) {
      }
    }
  }

  std::vector<uint64_t> samples;
  if (!is_server)
    samples.reserve(iters);

  if (is_server) {
    for (int i = 0; i < iters; i++) {
      while (poll_cq_one(r.cq).opcode != IBV_WC_RECV) {
      }
      post_recv(r, bytes, 1);
      post_send(r, bytes, 2);
      while (poll_cq_one(r.cq).opcode != IBV_WC_SEND) {
      }
    }
    return;
  }

  for (int i = 0; i < iters; i++) {
    uint64_t t0 = now_ns();
    post_send(r, bytes, 2);
    while (poll_cq_one(r.cq).opcode != IBV_WC_SEND) {
    }
    while (poll_cq_one(r.cq).opcode != IBV_WC_RECV) {
    }
    uint64_t t1 = now_ns();
    samples.push_back(t1 - t0);
    post_recv(r, bytes, 1);
  }

  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double q) -> uint64_t {
    size_t idx = (size_t)((sorted.size() - 1) * q);
    return sorted[idx];
  };
  long double avg = 0;
  for (auto ns : samples)
    avg += (long double)ns;
  avg /= (long double)samples.size();

  auto ns_to_us = [](uint64_t ns) { return (double)ns / 1000.0; };
  std::cout << "\n[Latency] message=" << bytes << "B, iters=" << iters << "\n";
  std::cout << "  RTT avg  = " << ns_to_us((uint64_t)avg) << " us\n";
  std::cout << "  RTT p50  = " << ns_to_us(pct(0.50)) << " us\n";
  std::cout << "  RTT p99  = " << ns_to_us(pct(0.99)) << " us\n";
  std::cout << "  One-way approx avg ~ " << ns_to_us((uint64_t)(avg / 2)) << " us\n";
}

// Bandwidth: RDMA WRITE (client -> server memory), then SEND 1B "done"
static void bandwidth_test_write(RdmaCtx &r, bool is_server, size_t total_bytes, uint32_t msg_bytes,
                                 int window, uint64_t remote_addr, uint32_t remote_rkey) {
  if (msg_bytes > r.buf_size)
    die("msg_bytes larger than buffer");
  if (window < 1)
    window = 1;
  window = std::min(window, std::max(1, r.qp_wr_limit / 16));

  std::cout << "[BW config] mode=RDMA_WRITE msg=" << msg_bytes << " window=" << window << "\n";

  const uint32_t done_bytes = 1;

  if (is_server) {
    // Only need one RECV for done signal
    post_recv(r, done_bytes, 5001);
    // wait DONE
    while (true) {
      ibv_wc wc = poll_cq_one(r.cq);
      if (wc.opcode == IBV_WC_RECV)
        break;
    }
    return;
  }

  if (remote_addr == 0 || remote_rkey == 0)
    die("Remote addr/rkey not provided by server");

  uint64_t t0 = now_ns();

  size_t sent = 0;
  int in_flight = 0;

  // Use remote_addr with offset to avoid always overwriting same cache line (optional)
  while (sent < total_bytes || in_flight > 0) {
    while (sent < total_bytes && in_flight < window) {
      uint32_t chunk = (uint32_t)std::min<size_t>(msg_bytes, total_bytes - sent);

      uint64_t roff = (sent % (size_t)r.buf_size);
      if (roff + chunk > r.buf_size)
        roff = 0; // wrap safely
      uint64_t raddr = remote_addr + roff;

      post_write(r, chunk, raddr, remote_rkey, 4000 + sent);
      sent += chunk;
      in_flight++;
    }

    ibv_wc wc = poll_cq_one(r.cq);
    if (wc.opcode == IBV_WC_RDMA_WRITE) {
      in_flight--;
    } else if (wc.opcode == IBV_WC_SEND) {
      // ignore
    } else if (wc.opcode == IBV_WC_RECV) {
      // ignore
    }
  }

  // Send DONE (1B)
  post_send(r, done_bytes, 6001);
  while (poll_cq_one(r.cq).opcode != IBV_WC_SEND) {
  }

  uint64_t t1 = now_ns();

  double sec = (double)(t1 - t0) / 1e9;
  double gib = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
  double gbps = (double)total_bytes * 8.0 / sec / 1e9;

  std::cout << "\n[Bandwidth] mode=RDMA_WRITE total=" << total_bytes << " msg=" << msg_bytes
            << " window=" << window << "\n";
  std::cout << "  Time = " << sec << " s\n";
  std::cout << "  Throughput = " << (gib / sec) << " GiB/s" << " (" << gbps << " Gb/s)\n";
}

static void usage() {
  std::cerr << "Usage:\n"
               "  --server --bind <ip> --local <local_ip> [--dev <dev>] [--port <n>] [--gid <idx>] "
               "[--msg <bytes>] [--total <bytes>] [--iters <n>] [--window <n>]\n"
               "  --client --connect <server_ip> --local <local_ip> [same opts]\n";
  std::exit(1);
}

int main(int argc, char **argv) {
  bool is_server = false, is_client = false;
  std::string bind_ip, connect_ip, dev_name, local_ip;
  int port = 1;
  int gid = -1;

  uint32_t msg_bytes = 1u << 20;
  size_t total_bytes = 100ull * 1024 * 1024;
  int iters = 10000;
  int window = 64;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char *opt) {
      if (i + 1 >= argc)
        die(std::string("Missing value for ") + opt);
      return std::string(argv[++i]);
    };

    if (a == "--server")
      is_server = true;
    else if (a == "--client")
      is_client = true;
    else if (a == "--bind")
      bind_ip = need("--bind");
    else if (a == "--connect")
      connect_ip = need("--connect");
    else if (a == "--dev")
      dev_name = need("--dev");
    else if (a == "--local")
      local_ip = need("--local");
    else if (a == "--port")
      port = std::stoi(need("--port"));
    else if (a == "--gid")
      gid = std::stoi(need("--gid"));
    else if (a == "--msg")
      msg_bytes = (uint32_t)std::stoul(need("--msg"));
    else if (a == "--total")
      total_bytes = (size_t)std::stoull(need("--total"));
    else if (a == "--iters")
      iters = std::stoi(need("--iters"));
    else if (a == "--window")
      window = std::stoi(need("--window"));
    else
      usage();
  }

  if (is_server == is_client)
    usage();
  if (is_server && bind_ip.empty())
    usage();
  if (is_client && connect_ip.empty())
    usage();
  if (local_ip.empty() && gid < 0)
    die("Need --local <ipv4> for auto-gid, or pass --gid explicitly.");

  std::cout << "Role: " << (is_server ? "SERVER" : "CLIENT") << "\n";

  // buffer must cover msg_bytes for RDMA write target, plus latency 64B
  size_t buf_size = std::max<size_t>(msg_bytes, 1024);

  // Server must allow remote write on its recv buffer MR
  bool allow_remote_write = is_server;

  RdmaCtx r = rdma_setup(dev_name, (uint8_t)port, gid, local_ip, buf_size, allow_remote_write);

  int sock = -1;
  if (is_server) {
    int lfd = tcp_listen(bind_ip);
    std::cout << "TCP control listening on " << bind_ip << ":" << kCtrlPort << "\n";
    sock = tcp_accept(lfd);
    ::close(lfd);
    std::cout << "TCP control accepted.\n";
  } else {
    sock = tcp_connect(connect_ip);
    std::cout << "TCP control connected to " << connect_ip << ":" << kCtrlPort << "\n";
  }

  CtrlMsg peer{};
  exchange_ctrl(sock, r, peer, is_server);

  connect_qp(r, peer);

  tcp_barrier(sock, 0xC0FFEE00);
  std::cout << "QP connected. Starting tests...\n";

  latency_test(r, is_server, iters);

  tcp_barrier(sock, 0xC0FFEE11);

  // For BW RDMA write: client needs server's raddr+rkey
  uint64_t remote_addr = peer.raddr;
  uint32_t remote_rkey = peer.rkey;

  bandwidth_test_write(r, is_server, total_bytes, msg_bytes, window, remote_addr, remote_rkey);

  ::close(sock);

  if (r.qp)
    ibv_destroy_qp(r.qp);
  if (r.cq)
    ibv_destroy_cq(r.cq);
  if (r.mr_send)
    ibv_dereg_mr(r.mr_send);
  if (r.mr_recv)
    ibv_dereg_mr(r.mr_recv);
  if (r.buf_send)
    std::free(r.buf_send);
  if (r.buf_recv)
    std::free(r.buf_recv);
  if (r.pd)
    ibv_dealloc_pd(r.pd);
  if (r.ctx)
    ibv_close_device(r.ctx);

  std::cout << "\nDone.\n";
  return 0;
}
