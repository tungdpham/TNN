#pragma once

#include <cstddef>
#include <map>
#include <mutex>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#ifndef NDEBUG
#include <iostream>
#endif

namespace tnn {

// Allocates a device pointer that contains a storage block that can be shared and automatically
// reclaimed by allocator by installing a custom deleter in device_storage's shared_ptr.
// Ensures user don't do some bad memory management.
class PoolAllocator : public IAllocator {
public:
  PoolAllocator(const Device &device)
      : device_(device) {}
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
    device_storage *ptr = allocate_storage(size);
    auto storage =
        std::shared_ptr<device_storage>(ptr, [this](device_storage *ptr) { this->reclaim(ptr); });
    return dptr(storage, 0, storage->capacity());
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &pair : free_blocks_) {
      delete pair.second;
    }
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
  std::multimap<size_t, device_storage *> free_blocks_;
  const Device &device_;
  mutable std::mutex mutex_;

  device_storage *allocate_storage(size_t size) {
    if (size == 0) {
      return new device_storage(device_);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_.lower_bound(size);

    if (it != free_blocks_.end()) {
      device_storage *block = it->second;
      if (block->capacity() <= size * 2) {
        free_blocks_.erase(it);
        return block;
      }
    }
#ifndef NDEBUG
    if (size > 0)
      std::cout << "PoolAllocator: Allocating new tensor of size " << size << " bytes.\n";
#endif

    void *ptr = device_.allocateAlignedMemory(size, DEFAULT_ALIGNMENT);
    return new device_storage(device_, ptr, size, DEFAULT_ALIGNMENT);
  }

  void reclaim(device_storage *storage) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.emplace(storage->capacity(), storage);
  }
};

}  // namespace tnn