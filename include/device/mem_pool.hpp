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
  MemPool() = default;
  ~MemPool() = default;

  MemPool(const MemPool &) = delete;
  MemPool &operator=(const MemPool &) = delete;

  device_ptr get(size_t size, const Device *device) {
    if (size == 0) {
      return make_dptr(device, 0);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_.lower_bound(size);

    while (it != free_blocks_.end()) {
      if (it->second.getDevice() == device) {
        device_ptr block = std::move(it->second);
        free_blocks_.erase(it);
        return block;
      }
      ++it;
    }
#ifndef NDEBUG
    if (size > 0)
      std::cout << "MemPool: Allocating new tensor of size " << size << " bytes.\n";
#endif

    return make_dptr(device, size);
  }

  void release(device_ptr &&ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t byte_size = ptr.capacity();
    free_blocks_.emplace(byte_size, std::move(ptr));
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.clear();
  }

private:
  std::multimap<size_t, device_ptr> free_blocks_;
  std::mutex mutex_;
};

inline MemPool &global_mem_pool() {
  static MemPool instance;
  return instance;
}

} // namespace tnn