#pragma once

#include <fcntl.h>
#include <infiniband/verbs.h>

#include <atomic>
#include <functional>

#include "asio/awaitable.hpp"
#include "asio/io_context.hpp"
#include "asio/posix/stream_descriptor.hpp"
#include "asio/use_awaitable.hpp"
#include "distributed/roce_device.hpp"

namespace tnn {
class RoCECQ {
public:
  RoCECQ(RoCEDevice &device, asio::io_context &io_context, int depth)
      : desc_(io_context) {
    context_ = device.get_context();
    comp_channel_ = ibv_create_comp_channel(context_);
    if (!comp_channel_) {
      throw std::runtime_error("Failed to create completion channel");
    }

    cq_ = ibv_create_cq(context_, depth, nullptr, comp_channel_, 0);
    if (!cq_) {
      ibv_destroy_comp_channel(comp_channel_);
      throw std::runtime_error("Failed to create completion queue");
    }

    int flags = fcntl(comp_channel_->fd, F_GETFL);
    fcntl(comp_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    desc_.assign(comp_channel_->fd);
    ibv_req_notify_cq(cq_, 0);
  }

  ~RoCECQ() {
    if (cq_) ibv_destroy_cq(cq_);
    if (comp_channel_) ibv_destroy_comp_channel(comp_channel_);
  }

  RoCECQ(const RoCECQ &) = delete;
  RoCECQ &operator=(const RoCECQ &) = delete;

  RoCECQ(RoCECQ &&other) noexcept
      : context_(other.context_),
        comp_channel_(other.comp_channel_),
        cq_(other.cq_),
        desc_(std::move(other.desc_)) {
    other.context_ = nullptr;
    other.comp_channel_ = nullptr;
    other.cq_ = nullptr;
  }

  RoCECQ &operator=(RoCECQ &&other) noexcept {
    if (this != &other) {
      if (cq_) ibv_destroy_cq(cq_);
      if (comp_channel_) ibv_destroy_comp_channel(comp_channel_);
      context_ = other.context_;
      comp_channel_ = other.comp_channel_;
      cq_ = other.cq_;
      desc_ = std::move(other.desc_);
      other.context_ = nullptr;
      other.comp_channel_ = nullptr;
      other.cq_ = nullptr;
    }
    return *this;
  }

  ibv_cq *handle() const { return cq_; }

  asio::awaitable<void> run_loop(std::atomic<bool> &is_running,
                                 std::function<void(ibv_wc *)> callback) {
    while (is_running.load(std::memory_order_acquire)) {
      co_await desc_.async_wait(asio::posix::stream_descriptor::wait_read, asio::use_awaitable);

      ibv_cq *ev_cq;
      void *ev_ctx;
      if (ibv_get_cq_event(comp_channel_, &ev_cq, &ev_ctx) == 0) {
        ibv_ack_cq_events(ev_cq, 1);
        ibv_req_notify_cq(ev_cq, 0);

        // Poll and execute callback for each Work Completion
        ibv_wc wc[16];
        int n;
        while ((n = ibv_poll_cq(cq_, 16, wc)) > 0) {
          for (int i = 0; i < n; i++) callback(&wc[i]);
        }
      }
    }
  }

private:
  ibv_context *context_;
  ibv_comp_channel *comp_channel_;
  ibv_cq *cq_ = nullptr;
  asio::posix::stream_descriptor desc_;
};
}  // namespace tnn