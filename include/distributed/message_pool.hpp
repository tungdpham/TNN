#pragma once

#include "message.hpp"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>

namespace tnn {
class MessagePool;

class MessageDeleter {
public:
  explicit MessageDeleter(MessagePool *pool = nullptr) : pool_(pool) {}

  void operator()(Message *ptr) const;

private:
  MessagePool *pool_;
};

using PooledMessage = std::unique_ptr<Message, MessageDeleter>;

class MessagePool {
public:
  static constexpr size_t MAX_POOL_SIZE = 64;

  PooledMessage get_message(size_t min_size, CommandType cmd_type) {
    MessageDeleter deleter(this);

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      return PooledMessage(new Message("", cmd_type), deleter);
    }

    {
      std::lock_guard<std::mutex> lock(pool_mutex_);

      auto it = pool_.begin();
      while (it != pool_.end()) {
        Message *raw_msg = *it;

        // Reuse messages that match command type and have sufficient size.
        // We also check upper bound to avoid wasting large messages on small requests,
        // similar to BufferPool logic.
        if (raw_msg->header().command_type == cmd_type && raw_msg->size() >= min_size &&
            raw_msg->size() <= min_size * 2) {

          pool_.erase(it);
          return PooledMessage(raw_msg, deleter);
        }
        ++it;
      }
    }

    return PooledMessage(new Message("", cmd_type), deleter);
  }

  static MessagePool &instance() {
    static MessagePool pool;
    return pool;
  }

  ~MessagePool() {
    is_shutting_down_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (Message *msg : pool_) {
      delete msg;
    }
    pool_.clear();
  }

  void return_message_internal(Message *message) {
    if (message == nullptr) {
      return;
    }

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      delete message;
      return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (pool_.size() < MAX_POOL_SIZE) {
      pool_.push_back(message);
    } else {
      delete message;
    }
  }

  bool is_shutting_down() const { return is_shutting_down_.load(std::memory_order_relaxed); }

  size_t pool_size() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return pool_.size();
  }

private:
  std::atomic<bool> is_shutting_down_{false};
  std::deque<Message *> pool_;
  mutable std::mutex pool_mutex_;
};

inline void MessageDeleter::operator()(Message *ptr) const {
  if (pool_) {
    pool_->return_message_internal(ptr);
  } else {
    delete ptr;
  }
}

} // namespace tnn