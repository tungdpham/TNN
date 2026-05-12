#pragma once

#include <ucp/api/ucp.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <asio.hpp>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "communicator.hpp"
#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/io.hpp"
#include "message.hpp"

namespace tnn {

struct UCXIdentityMessage {
  int32_t listening_port;
};

template <typename Archiver>
void archive(Archiver &archiver, const UCXIdentityMessage &identity) {
  archiver(identity.listening_port);
}

template <typename Archiver>
void archive(Archiver &archiver, UCXIdentityMessage &identity) {
  archiver(identity.listening_port);
}

struct UCXReq {
  bool done = false;
  ucs_status_t status = UCS_OK;
};

inline void ucx_send_cb(void *request, ucs_status_t status) {
  auto *r = reinterpret_cast<UCXReq *>(request);
  r->status = status;
  r->done = true;
}

inline void ucx_recv_cb(void *request, ucs_status_t status, ucp_tag_recv_info_t *) {
  auto *r = reinterpret_cast<UCXReq *>(request);
  r->status = status;
  r->done = true;
}

struct UCXEndpointHash {
  size_t operator()(const Endpoint &endpoint) const {
    return endpoint.hash();
  }
};

class UCXChannel {
public:
  UCXChannel(ucp_worker_h worker, ucp_ep_h ep, Endpoint endpoint, uint64_t tag_base)
      : worker(worker), ep(ep), endpoint(std::move(endpoint)), tag_base(tag_base) {}

  ucp_worker_h worker = nullptr;
  ucp_ep_h ep = nullptr;
  Endpoint endpoint;
  uint64_t tag_base = 0;
  std::atomic<uint64_t> send_seq{1};
  std::atomic<uint64_t> recv_seq{1};
  std::mutex send_mutex;
};

class UCXCommunicator : public Communicator {
public:
  struct Config {
    uint32_t num_io_threads = 4;
  };

  static std::unique_ptr<UCXCommunicator> create(const Endpoint &endpoint,
                                                Config config) {
    auto &alloc = PoolAllocator::instance(getHost(), defaultFlowHandle);
    auto comm = std::make_unique<UCXCommunicator>(endpoint, alloc, config);
    comm->start_server();
    return comm;
  }

  explicit UCXCommunicator(const Endpoint &endpoint, IAllocator &out_allocator,
                           Config config)
      : Communicator(endpoint, config.num_io_threads),
        int_allocator_(PoolAllocator::instance(getHost(), defaultFlowHandle)),
        out_allocator_(out_allocator),
        serializer_(int_allocator_),
        config_(config),
        acceptor_(io_context_pool_.acceptor()) {
    init_ucx();
  }

  ~UCXCommunicator() override { stop(); }

  void start_server() {
    int port = endpoint_.get_parameter<int>("port");

    asio::ip::tcp::endpoint ep(asio::ip::tcp::v4(), static_cast<asio::ip::port_type>(port));
    acceptor_.open(ep.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(ep);
    acceptor_.listen();

    is_running_.store(true, std::memory_order_release);

    asio::co_spawn(acceptor_.get_executor(), [this]() { return accept_loop(); }, asio::detached);

    io_thread_ = std::thread([this]() { io_context_pool_.run(); });
    progress_thread_ = std::thread([this]() { progress_loop(); });

    std::cout << "[UCX] server started at " << endpoint_.id() << std::endl;
  }

  void stop() {
    if (!is_running_.exchange(false)) return;

    std::error_code ec;
    if (acceptor_.is_open()) acceptor_.close(ec);

    {
      std::lock_guard<std::shared_mutex> lock(channels_mutex_);
      for (auto &kv : channels_) {
        if (kv.second->ep) {
          {
            std::lock_guard<std::mutex> lk(ucx_mutex_);
            ucp_ep_destroy(kv.second->ep);
          }
          kv.second->ep = nullptr;
        }
      }
      channels_.clear();
    }

    io_context_pool_.stop();
    if (io_thread_.joinable()) io_thread_.join();
    if (progress_thread_.joinable()) progress_thread_.join();

    if (worker_) {
      ucp_worker_destroy(worker_);
      worker_ = nullptr;
    }
    if (context_) {
      ucp_cleanup(context_);
      context_ = nullptr;
    }
  }

  void send_impl(Message &&message, const Endpoint &endpoint) override {
    try {
      std::shared_ptr<UCXChannel> ch;
      {
        std::shared_lock<std::shared_mutex> lock(channels_mutex_);
        auto it = channels_.find(endpoint);
        if (it != channels_.end()) ch = it->second;
      }
      if (!ch) {
        std::cerr << "[UCX] no channel to endpoint " << endpoint.id() << std::endl;
        return;
      }

      auto dbg_cmd = message.header().command_type;

      Sizer sizer;
      sizer(message);
      size_t msg_size = sizer.size();

      std::cout << "[UCXDBG][send]"
                << " to=" << endpoint.id()
                << " cmd=" << static_cast<int>(dbg_cmd)
                << " bytes=" << msg_size
                << std::endl;

      dptr buffer = int_allocator_.allocate(msg_size);
      Writer writer(buffer);
      writer(message);

      uint64_t seq = ch->send_seq.fetch_add(1);
      uint64_t len = msg_size;

      std::lock_guard<std::mutex> send_lock(ch->send_mutex);

      send_blocking(ch, &len, sizeof(len), make_tag(ch->tag_base, seq, 0));
      send_blocking(ch, buffer.get(), msg_size, make_tag(ch->tag_base, seq, 1));

      add_profile_data("ucx_send_msg_count", 1);
      add_profile_data("ucx_send_bytes", static_cast<int64_t>(msg_size));
    } catch (const std::exception &e) {
      std::cerr << "[UCX] send error: " << e.what() << std::endl;
    }
  }

  void flush_output_messages() override {
    std::unique_lock<std::mutex> lock(this->out_message_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || this->out_message_queue_.empty()) return;

    while (!this->out_message_queue_.empty()) {
      auto [msg, endpoint] = std::move(this->out_message_queue_.front());
      this->out_message_queue_.pop();
      send_impl(std::move(msg), endpoint);
    }
  }

  IAllocator &out_allocator() override { return out_allocator_; }

  bool connect_to_endpoint(const Endpoint &endpoint) override {
    if (endpoint.type() == CommunicationType::IN_PROCESS) {
      return true;
    }
    try {
      std::string host = endpoint.get_parameter<std::string>("host");
      int port = endpoint.get_parameter<int>("port");

      asio::io_context ctx;
      asio::ip::tcp::socket sock(ctx);
      asio::ip::tcp::resolver resolver(ctx);
      asio::connect(sock, resolver.resolve(host, std::to_string(port)));

      UCXIdentityMessage identity{endpoint_.get_parameter<int>("port")};
      send_identity(sock, identity);

      auto ch = create_channel_via_socket(sock, endpoint);

      {
        std::lock_guard<std::shared_mutex> lock(channels_mutex_);
        channels_[endpoint] = ch;
      }

      std::thread([this, ch]() { recv_loop(ch); }).detach();

      std::cout << "[UCX] connected to " << endpoint.id() << std::endl;
      return true;
    } catch (const std::exception &e) {
      std::cerr << "[UCX] connect error: " << e.what() << std::endl;
      return false;
    }
  }

  bool disconnect_from_endpoint(const Endpoint &endpoint) override {
    std::lock_guard<std::shared_mutex> lock(channels_mutex_);
    auto it = channels_.find(endpoint);
    if (it != channels_.end()) {
      if (it->second->ep) {
        std::lock_guard<std::mutex> lk(ucx_mutex_);
        ucp_ep_destroy(it->second->ep);
      }
      channels_.erase(it);
    }
    return true;
  }

private:
  void init_ucx() {
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE;
    params.features = UCP_FEATURE_TAG;
    params.request_size = sizeof(UCXReq);

    ucp_config_t *config = nullptr;
    if (ucp_config_read(nullptr, nullptr, &config) != UCS_OK) {
      throw std::runtime_error("ucp_config_read failed");
    }

    ucs_status_t st = ucp_init(&params, config, &context_);
    ucp_config_release(config);
    if (st != UCS_OK) {
      throw std::runtime_error(std::string("ucp_init failed: ") + ucs_status_string(st));
    }

    ucp_worker_params_t wparams{};
    wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SERIALIZED;

    st = ucp_worker_create(context_, &wparams, &worker_);
    if (st != UCS_OK) {
      throw std::runtime_error(std::string("ucp_worker_create failed: ") + ucs_status_string(st));
    }
  }

  static uint64_t make_tag(uint64_t tag_base, uint64_t seq, uint64_t part) {
    (void)tag_base;
    return ((seq & 0x00FFFFFFFFFFFFFFULL) << 1) | (part & 1ULL);
  }

  void progress_loop() {
    while (is_running_.load(std::memory_order_acquire)) {
      {
        std::lock_guard<std::mutex> lk(ucx_mutex_);
        ucp_worker_progress(worker_);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  void wait_req(void *req) {
    if (req == nullptr) return;
    if (UCS_PTR_IS_ERR(req)) {
      throw std::runtime_error(std::string("UCX request error: ") +
                               ucs_status_string(UCS_PTR_STATUS(req)));
    }

    UCXReq *r = reinterpret_cast<UCXReq *>(req);
    r->done = false;
    r->status = UCS_OK;

    while (!r->done) {
      std::lock_guard<std::mutex> lk(ucx_mutex_);
      ucp_worker_progress(worker_);
    }

    if (r->status != UCS_OK) {
      std::string err = ucs_status_string(r->status);
      ucp_request_free(req);
      throw std::runtime_error("UCX op failed: " + err);
    }

    ucp_request_free(req);
  }

  void send_blocking(std::shared_ptr<UCXChannel> ch, const void *buf, size_t bytes, uint64_t tag) {
    void *req = nullptr;
    {
      std::lock_guard<std::mutex> lk(ucx_mutex_);
      req = ucp_tag_send_nb(ch->ep, const_cast<void *>(buf), bytes, ucp_dt_make_contig(1),
                            tag, ucx_send_cb);
    }
    wait_req(req);
  }

  void recv_blocking(void *buf, size_t bytes, uint64_t tag) {
    void *req = nullptr;
    {
      std::lock_guard<std::mutex> lk(ucx_mutex_);
      req = ucp_tag_recv_nb(worker_, buf, bytes, ucp_dt_make_contig(1), tag, UINT64_MAX,
                            ucx_recv_cb);
    }
    wait_req(req);
  }

  void send_raw(asio::ip::tcp::socket &sock, const void *data, size_t n) {
    asio::write(sock, asio::buffer(data, n));
  }

  void recv_raw(asio::ip::tcp::socket &sock, void *data, size_t n) {
    asio::read(sock, asio::buffer(data, n));
  }

  void send_identity(asio::ip::tcp::socket &sock, const UCXIdentityMessage &identity) {
    Sizer sizer;
    sizer(identity);
    size_t n = sizer.size();
    dptr buf = int_allocator_.allocate(n);
    Writer writer(buf);
    writer(identity);
    send_raw(sock, &n, sizeof(n));
    send_raw(sock, buf.get(), n);
  }

  UCXIdentityMessage recv_identity(asio::ip::tcp::socket &sock) {
    size_t n = 0;
    recv_raw(sock, &n, sizeof(n));
    dptr buf = int_allocator_.allocate(n);
    recv_raw(sock, buf.get(), n);
    Reader reader(buf);
    UCXIdentityMessage identity;
    reader(identity);
    return identity;
  }

  std::shared_ptr<UCXChannel> create_channel_via_socket(asio::ip::tcp::socket &sock,
                                                        const Endpoint &peer_endpoint) {
    ucp_address_t *local_addr = nullptr;
    size_t local_len = 0;
    {
      std::lock_guard<std::mutex> lk(ucx_mutex_);
      if (ucp_worker_get_address(worker_, &local_addr, &local_len) != UCS_OK) {
        throw std::runtime_error("ucp_worker_get_address failed");
      }
    }

    uint64_t my_len = local_len;
    send_raw(sock, &my_len, sizeof(my_len));
    send_raw(sock, local_addr, local_len);

    uint64_t peer_len = 0;
    recv_raw(sock, &peer_len, sizeof(peer_len));
    std::vector<char> peer_addr(peer_len);
    recv_raw(sock, peer_addr.data(), peer_len);

    {
      std::lock_guard<std::mutex> lk(ucx_mutex_);
      ucp_worker_release_address(worker_, local_addr);
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t *>(peer_addr.data());

    ucp_ep_h ep = nullptr;
    ucs_status_t st;
    {
      std::lock_guard<std::mutex> lk(ucx_mutex_);
      st = ucp_ep_create(worker_, &ep_params, &ep);
    }
    if (st != UCS_OK) {
      throw std::runtime_error(std::string("ucp_ep_create failed: ") + ucs_status_string(st));
    }

    uint64_t local_tag_id =
        (static_cast<uint64_t>(endpoint_.get_parameter<int>("port")) << 32) ^
        channel_id_counter_.fetch_add(1);

    send_raw(sock, &local_tag_id, sizeof(local_tag_id));
    uint64_t peer_tag_id = 0;
    recv_raw(sock, &peer_tag_id, sizeof(peer_tag_id));

    uint64_t tag_base = ((local_tag_id ^ peer_tag_id) & 0x7FULL) << 56;

    std::cout << "[UCX] channel tag_base=0x"
              << std::hex << tag_base << std::dec
              << " peer=" << peer_endpoint.id()
              << std::endl;

    return std::make_shared<UCXChannel>(worker_, ep, peer_endpoint, tag_base);
  }

  asio::awaitable<void> accept_loop() {
    while (is_running_.load(std::memory_order_acquire)) {
      try {
        asio::ip::tcp::socket sock(co_await asio::this_coro::executor);
        co_await acceptor_.async_accept(sock, asio::use_awaitable);

        auto identity = recv_identity(sock);

        std::string host = sock.remote_endpoint().address().to_string();
        Endpoint peer = Endpoint::ucx(host, identity.listening_port);

        auto ch = create_channel_via_socket(sock, peer);

        {
          std::lock_guard<std::shared_mutex> lock(channels_mutex_);
          channels_[peer] = ch;
        }

        std::thread([this, ch]() { recv_loop(ch); }).detach();

        std::cout << "[UCX] accepted peer " << peer.id() << std::endl;
      } catch (const std::exception &e) {
        if (is_running_.load()) {
          std::cerr << "[UCX] accept error: " << e.what() << std::endl;
        }
      }
    }
  }

  void recv_loop(std::shared_ptr<UCXChannel> ch) {
    while (is_running_.load(std::memory_order_acquire)) {
      try {
        uint64_t seq = ch->recv_seq.fetch_add(1);
        uint64_t len = 0;

        recv_blocking(&len, sizeof(len), make_tag(ch->tag_base, seq, 0));
        if (len == 0 || len > (2ULL << 30)) {
          std::cerr << "[UCX] invalid message length " << len << std::endl;
          continue;
        }

        dptr buf = int_allocator_.allocate(len);
        recv_blocking(buf.get(), len, make_tag(ch->tag_base, seq, 1));

        Reader reader(buf);
        Message msg;
        serializer_.deserialize(reader, msg);

        auto dbg_cmd = msg.header().command_type;
        std::cout << "[UCXDBG][recv]"
                  << " from=" << ch->endpoint.id()
                  << " cmd=" << static_cast<int>(dbg_cmd)
                  << " bytes=" << len
                  << std::endl;

        this->enqueue_input_message(std::move(msg));

        add_profile_data("ucx_recv_msg_count", 1);
        add_profile_data("ucx_recv_bytes", static_cast<int64_t>(len));
      } catch (const std::exception &e) {
        if (is_running_.load()) {
          std::cerr << "[UCX] recv error from " << ch->endpoint.id() << ": " << e.what()
                    << std::endl;
        }
        break;
      }
    }
  }

private:
  IAllocator &int_allocator_;
  IAllocator &out_allocator_;
  BinarySerializer serializer_;
  Config config_;

  ucp_context_h context_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  std::mutex ucx_mutex_;

  asio::ip::tcp::acceptor acceptor_;
  std::atomic<bool> is_running_{false};
  std::thread io_thread_;
  std::thread progress_thread_;

  std::atomic<uint64_t> channel_id_counter_{1};
  std::shared_mutex channels_mutex_;
  std::unordered_map<Endpoint, std::shared_ptr<UCXChannel>, UCXEndpointHash> channels_;
};

}  // namespace tnn
