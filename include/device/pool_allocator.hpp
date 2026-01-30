#pragma once

#include "device/allocator.hpp"
#include "device/dptr.hpp"

#include <cstddef>
#include <map>
#include <mutex>
#ifndef NDEBUG
#include <iostream>
#endif

namespace tnn {

class PoolAllocator : public IAllocator {
public:
  PoolAllocator(const Device &device) : device_(device) {}
  ~PoolAllocator() = default;

  PoolAllocator(const PoolAllocator &) = delete;
  PoolAllocator &operator=(const PoolAllocator &) = delete;

  static PoolAllocator &instance(const Device &device) {
    static std::mutex registry_mutex;
    static std::map<const Device *, std::unique_ptr<PoolAllocator>> instances;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto &pool = instances[&device];
    if (!pool) {
      pool = std::make_unique<PoolAllocator>(device);
    }
    return *pool;
  }

  dptr allocate(size_t size) override {
    if (size == 0) {
      return make_dptr(&device_, 0);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_.lower_bound(size);

    while (it != free_blocks_.end()) {
      dptr block = std::move(it->second);
      free_blocks_.erase(it);
      return block;
      ++it;
    }
#ifndef NDEBUG
    if (size > 0)
      std::cout << "PoolAllocator: Allocating new tensor of size " << size << " bytes.\n";
#endif

    return make_dptr(&device_, size);
  }

  void deallocate(dptr &&ptr) override {
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
  std::multimap<size_t, dptr> free_blocks_;
  const Device &device_;
  mutable std::mutex mutex_;
};

} // namespace tnn