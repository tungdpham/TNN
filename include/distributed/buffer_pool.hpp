#pragma once

#include <deque>
#include <memory>
#include <mutex>

#include "tbuffer.hpp"

namespace tnn {
class BufferPool;

class BufferDeleter {
public:
  explicit BufferDeleter(BufferPool *pool = nullptr) : pool_(pool) {}

  void operator()(TBuffer *ptr) const;

private:
  BufferPool *pool_;
};

using PooledBuffer = std::shared_ptr<TBuffer>;

class BufferPool {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;
  static constexpr size_t MAX_POOL_SIZE = 128;

  PooledBuffer get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    BufferDeleter deleter(this);

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      return PooledBuffer(new TBuffer(min_size), deleter);
    }

    {
      std::lock_guard<std::mutex> lock(pool_mutex_);

      auto it = pool_.begin();
      while (it != pool_.end()) {
        TBuffer *raw_buf = *it;

        // Reuse buffers within 2x of requested size to reduce allocations
        if (raw_buf->capacity() >= min_size && raw_buf->capacity() <= min_size * 2) {
          pool_.erase(it);
          raw_buf->clear();
          return PooledBuffer(raw_buf, deleter);
        }
        ++it;
      }
    }

    return PooledBuffer(new TBuffer(min_size), deleter);
  }

  static BufferPool &instance() {
    static BufferPool pool;
    return pool;
  }

  ~BufferPool() {
    is_shutting_down_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (TBuffer *buf : pool_) {
      delete buf;
    }
    pool_.clear();
  }

  void return_buffer_internal(TBuffer *buffer) {
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

  bool is_shutting_down() const { return is_shutting_down_.load(std::memory_order_relaxed); }

  size_t pool_size() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return pool_.size();
  }

private:
  std::atomic<bool> is_shutting_down_{false};
  std::deque<TBuffer *> pool_;
  mutable std::mutex pool_mutex_;
};

inline void BufferDeleter::operator()(TBuffer *ptr) const {
  if (pool_) {
    pool_->return_buffer_internal(ptr);
  } else {
    delete ptr;
  }
}

}  // namespace tnn