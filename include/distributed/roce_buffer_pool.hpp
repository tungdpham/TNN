/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>

#include "roce_buffer.hpp"

namespace tnn {

class RoceBufferPool;

class RoceBufferDeleter {
public:
  explicit RoceBufferDeleter(RoceBufferPool *pool = nullptr) : pool_(pool) {}
  void operator()(RoceBuffer *ptr) const;

private:
  RoceBufferPool *pool_;
};

using PooledRoceBuffer = std::shared_ptr<RoceBuffer>;

class RoceBufferPool {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;
  static constexpr size_t MAX_POOL_SIZE = 128;

  explicit RoceBufferPool(ibv_pd *pd) : pd_(pd) {}

  PooledRoceBuffer get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    RoceBufferDeleter deleter(this);

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      return PooledRoceBuffer(new RoceBuffer(pd_, min_size), deleter);
    }

    {
      std::lock_guard<std::mutex> lock(pool_mutex_);

      auto it = pool_.begin();
      while (it != pool_.end()) {
        RoceBuffer *raw_buf = *it;

        if (raw_buf->capacity() >= min_size && raw_buf->capacity() <= min_size * 2) {
          pool_.erase(it);
          raw_buf->clear();
          return PooledRoceBuffer(raw_buf, deleter);
        }
        ++it;
      }
    }

    return PooledRoceBuffer(new RoceBuffer(pd_, min_size), deleter);
  }

  ~RoceBufferPool() {
    is_shutting_down_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (RoceBuffer *buf : pool_) {
      delete buf;
    }
    pool_.clear();
  }

  void return_buffer_internal(RoceBuffer *buffer) {
    if (buffer == nullptr) {
      return;
    }

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      delete buffer;
      return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (pool_.size() < MAX_POOL_SIZE) {
      buffer->clear();
      pool_.push_back(buffer);
    } else {
      delete buffer;
    }
  }

private:
  ibv_pd *pd_;
  std::atomic<bool> is_shutting_down_{false};
  std::deque<RoceBuffer *> pool_;
  mutable std::mutex pool_mutex_;
};

inline void RoceBufferDeleter::operator()(RoceBuffer *ptr) const {
  if (pool_) {
    pool_->return_buffer_internal(ptr);
  } else {
    delete ptr;
  }
}

}  // namespace tnn
