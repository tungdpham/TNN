#pragma once

#include "device/device_ptr.hpp"

#include <cstddef>
#include <map>
#include <mutex>
#ifndef NDEBUG
#include <iostream>
#endif

namespace tnn {

class MemPool {
public:
  MemPool(const Device &device) : device_(device) {}
  ~MemPool() = default;

  MemPool(const MemPool &) = delete;
  MemPool &operator=(const MemPool &) = delete;

  static MemPool &instance(const Device &device) {
    static std::mutex registry_mutex;
    static std::map<const Device *, std::unique_ptr<MemPool>> instances;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto &pool = instances[&device];
    if (!pool) {
      pool = std::make_unique<MemPool>(device);
    }
    return *pool;
  }

  device_ptr get(size_t size) {
    if (size == 0) {
      return make_dptr(&device_, 0);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_.lower_bound(size);

    while (it != free_blocks_.end()) {
      device_ptr block = std::move(it->second);
      free_blocks_.erase(it);
      return block;
      ++it;
    }
#ifndef NDEBUG
    if (size > 0)
      std::cout << "MemPool: Allocating new tensor of size " << size << " bytes.\n";
#endif

    return make_dptr(&device_, size);
  }

  void release(device_ptr &&ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t byte_size = ptr.capacity();
    if (byte_size == 0) {
      return;
    }
    free_blocks_.emplace(byte_size, std::move(ptr));
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.clear();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
  }

  size_t cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_) {
      total += pair.first;
    }
    return total;
  }

  const Device &device() const { return device_; }

private:
  std::multimap<size_t, device_ptr> free_blocks_;
  const Device &device_;
  mutable std::mutex mutex_;
};

} // namespace tnn