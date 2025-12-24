#pragma once

#include "tbuffer.hpp"
#include <deque>

namespace tnn {
class BufferPool;

class BufferDeleter {
public:
  explicit BufferDeleter(BufferPool *pool = nullptr) : pool_(pool) {}

  void operator()(TBuffer *ptr) const;

private:
  BufferPool *pool_;
};

using PooledBuffer = std::unique_ptr<TBuffer, BufferDeleter>;

class BufferPool {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;
  static constexpr size_t MAX_POOL_SIZE = 32;

  PooledBuffer get_buffer(size_t min_size = DEFAULT_BUFFER_SIZE) {
    BufferDeleter deleter(this);

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      return PooledBuffer(new TBuffer(min_size), deleter);
    }

    thread_local std::deque<TBuffer *> local_pool;

    while (!local_pool.empty()) {
      TBuffer *raw_buf = local_pool.front();
      local_pool.pop_front();

      if (raw_buf->capacity() >= min_size) {
        raw_buf->clear();

        return PooledBuffer(raw_buf, deleter);
      }

      delete raw_buf;
    }

    return PooledBuffer(new TBuffer(min_size), deleter);
  }

  static BufferPool &instance() {
    static BufferPool pool;
    return pool;
  }

  ~BufferPool() { is_shutting_down_.store(true, std::memory_order_release); }

  void return_buffer_internal(TBuffer *buffer) {
    if (buffer == nullptr) {
      return;
    }

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      delete buffer;
      return;
    }

    thread_local std::deque<TBuffer *> local_pool;
    if (local_pool.size() < MAX_POOL_SIZE) {
      buffer->clear();
      local_pool.push_back(buffer);
    } else {
      delete buffer;
    }
  }

  bool is_shutting_down() const { return is_shutting_down_.load(std::memory_order_relaxed); }

private:
  std::atomic<bool> is_shutting_down_{false};
};

inline void BufferDeleter::operator()(TBuffer *ptr) const {
  if (pool_) {
    pool_->return_buffer_internal(ptr);
  } else {
    delete ptr;
  }
}

} // namespace tnn