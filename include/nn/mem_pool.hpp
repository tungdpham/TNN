#pragma once

#include "tensor/tensor.hpp"

#include <cstddef>
#include <map>
#include <mutex>
#include <numeric>

namespace tnn {
template <typename T> class MemPool {
public:
  MemPool() = default;
  ~MemPool() = default;

  MemPool(const MemPool &) = delete;
  MemPool &operator=(const MemPool &) = delete;

  Tensor<T> get(const std::vector<size_t> &shape, const Device *device) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t required_bytes =
        std::accumulate(shape.begin(), shape.end(), sizeof(T), std::multiplies<size_t>());

    auto it = free_blocks_.lower_bound(required_bytes);

    while (it != free_blocks_.end()) {
      if (it->second.data.device() == device) {
        Tensor<T> found_tensor = std::move(it->second.data);
        free_blocks_.erase(it);

        found_tensor.ensure(shape);
        return found_tensor;
      }
      ++it;
    }

    return Tensor<T>(shape, device);
  }

  void release(Tensor<T> &&tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t type_size = tensor.capacity() * sizeof(T);
    MemBlock block{std::move(tensor)};
    free_blocks_.emplace(type_size, std::move(block));
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.clear();
  }

private:
  struct MemBlock {
    Tensor<T> data;
  };

  std::multimap<size_t, MemBlock> free_blocks_;
  std::mutex mutex_;
};

template <typename T> class TempBuffer {
public:
  TempBuffer(MemPool<T> &mem_pool, std::vector<size_t> shape, const Device *device)
      : mem_pool_(mem_pool) {
    buffer_ = mem_pool_.get(shape, device);
  }

  TempBuffer(const TempBuffer &) = delete;
  TempBuffer &operator=(const TempBuffer &) = delete;

  TempBuffer(TempBuffer &&other) noexcept
      : mem_pool_(other.mem_pool_), buffer_(std::move(other.buffer_)) {}

  TempBuffer &operator=(TempBuffer &&other) noexcept {
    if (this != &other) {
      if (buffer_.capacity() > 0) {
        mem_pool_.release(std::move(buffer_));
      }
      buffer_ = std::move(other.buffer_);
    }
    return *this;
  }

  ~TempBuffer() {
    if (buffer_.capacity() > 0) {
      mem_pool_.release(std::move(buffer_));
    }
  }

  Tensor<T> &get() { return buffer_; }

private:
  MemPool<T> &mem_pool_;
  Tensor<T> buffer_;
};

// Singleton accessor for default memory pool
template <typename T> inline MemPool<T> &getDefaultMemPool() {
  static MemPool<T> instance;
  return instance;
}

} // namespace tnn