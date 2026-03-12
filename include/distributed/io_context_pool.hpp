#pragma once

#include <asio.hpp>
#include <memory>
#include <thread>
#include <vector>

namespace tnn {
class IoContextPool {
public:
  explicit IoContextPool(std::size_t pool_size)
      : next_io_context_(0) {
    if (pool_size == 0) pool_size = 1;

    for (std::size_t i = 0; i < pool_size; ++i) {
      io_contexts_.emplace_back(std::make_shared<asio::io_context>());
      work_guards_.emplace_back(asio::make_work_guard(*io_contexts_.back()));
    }
  }

  void run() {
    std::vector<std::thread> threads;
    for (auto &ctx : io_contexts_) {
      threads.emplace_back([ctx]() { ctx->run(); });
    }
    for (auto &t : threads) t.join();
  }

  void stop() {
    for (auto &ctx : io_contexts_) ctx->stop();
  }

  // Round-robin assignment
  asio::io_context &get() {
    asio::io_context &io_context = *io_contexts_[next_io_context_];
    ++next_io_context_;
    if (next_io_context_ == io_contexts_.size()) next_io_context_ = 0;
    return io_context;
  }

  asio::io_context &acceptor() { return *io_contexts_[0]; }

private:
  std::vector<std::shared_ptr<asio::io_context>> io_contexts_;
  std::vector<asio::executor_work_guard<asio::io_context::executor_type>> work_guards_;
  std::size_t next_io_context_;
};

}  // namespace tnn