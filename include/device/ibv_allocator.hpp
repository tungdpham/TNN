/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/allocator.hpp"
#include "device/dptr.hpp"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <infiniband/verbs.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#ifndef NDEBUG
#include <iostream>
#endif

namespace tnn {

// InfiniBand Verbs allocator that manages a large master slab and slices it into individual
// buffers (dptrs) for tensor data. Each dptr has a custom deleter that returns the memory
// back to the allocator's free list.
class IbvAllocator : public IAllocator {
public:
  IbvAllocator(const Device &device, ibv_pd *pd, size_t slab_size)
      : device_(device), pd_(pd), slab_size_(slab_size), allocated_(0), using_host_memory_(false) {
    if (!pd_) {
      throw std::invalid_argument("Protection Domain cannot be null");
    }
    if (slab_size_ == 0) {
      throw std::invalid_argument("Slab size must be greater than 0");
    }

    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    // Try GPU Direct RDMA first (allocate on device and register with InfiniBand)
    slab_ptr_ = device_.allocateAlignedMemory(slab_size_, DEFAULT_ALIGNMENT);
    if (!slab_ptr_) {
      throw std::runtime_error("Failed to allocate master slab");
    }

    slab_mr_ = ibv_reg_mr(pd_, slab_ptr_, slab_size_, access_flags);
    if (!slab_mr_) {
      int err = errno;
#ifndef NDEBUG
      std::cout << "IbvAllocator: GPU Direct RDMA registration failed: " << std::strerror(err)
                << " (errno=" << err << "). Falling back to host pinned memory.\n";
#endif
      // Free GPU memory
      device_.deallocateAlignedMemory(slab_ptr_);
      slab_ptr_ = nullptr;

      // Fallback to host pinned memory
      if (posix_memalign(&slab_ptr_, DEFAULT_ALIGNMENT, slab_size_) != 0) {
        throw std::runtime_error("Failed to allocate host pinned memory");
      }

      // Register host memory with InfiniBand
      slab_mr_ = ibv_reg_mr(pd_, slab_ptr_, slab_size_, access_flags);
      if (!slab_mr_) {
        err = errno;
        free(slab_ptr_);
        slab_ptr_ = nullptr;
        throw std::runtime_error(
            "Failed to register host memory with InfiniBand: " + std::string(std::strerror(err)) +
            " (errno=" + std::to_string(err) + ")");
      }

      using_host_memory_ = true;
#ifndef NDEBUG
      std::cout << "IbvAllocator: Successfully registered host pinned memory of " << slab_size_
                << " bytes at " << slab_ptr_ << " with lkey=" << slab_mr_->lkey << "\n";
#endif
    } else {
#ifndef NDEBUG
      std::cout << "IbvAllocator: Registered GPU memory slab of " << slab_size_ << " bytes at "
                << slab_ptr_ << " with lkey=" << slab_mr_->lkey << " (GPU Direct RDMA)\n";
#endif
    }
  }

  ~IbvAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Deregister the memory region
    if (slab_mr_) {
      ibv_dereg_mr(slab_mr_);
      slab_mr_ = nullptr;
    }

    // Free the master slab
    if (slab_ptr_) {
      if (using_host_memory_) {
        free(slab_ptr_);
      } else {
        device_.deallocateAlignedMemory(slab_ptr_);
      }
      slab_ptr_ = nullptr;
    }

    free_blocks_.clear();
  }

  IbvAllocator(const IbvAllocator &) = delete;
  IbvAllocator &operator=(const IbvAllocator &) = delete;

  static IbvAllocator &instance(const Device &device, ibv_pd *pd, size_t slab_size) {
    static std::mutex registry_mutex;
    static std::map<const Device *, std::unique_ptr<IbvAllocator>> instances;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto &allocator = instances[&device];
    if (!allocator) {
      allocator = std::make_unique<IbvAllocator>(device, pd, slab_size);
    }
    return *allocator;
  }

  dptr allocate(size_t size) override {
    if (size == 0) {
      return dptr(nullptr);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_.lower_bound(size);

    if (it != free_blocks_.end()) {
      size_t offset = it->second;
      size_t block_size = it->first;
      free_blocks_.erase(it);

#ifndef NDEBUG
      std::cout << "IbvAllocator: Reusing block of size " << block_size << " at offset " << offset
                << "\n";
#endif

      return create_dptr(offset, block_size);
    }

    if (allocated_ + size > slab_size_) {
      throw std::runtime_error("IbvAllocator: Out of memory in master slab");
    }

    size_t offset = allocated_;
    allocated_ += size;

#ifndef NDEBUG
    std::cout << "IbvAllocator: Allocating new block of size " << size << " at offset " << offset
              << " (total allocated: " << allocated_ << " / " << slab_size_ << ")\n";
#endif

    return create_dptr(offset, size);
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.clear();
    allocated_ = 0;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
  }

  size_t allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;
  }

  size_t cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_) {
      total += pair.first;
    }
    return total;
  }

  size_t slab_size() const { return slab_size_; }

  const Device &device() const { return device_; }

  ibv_mr *get_mr() const { return slab_mr_; }

  uint32_t get_lkey() const { return slab_mr_ ? slab_mr_->lkey : 0; }

  uint32_t get_rkey() const { return slab_mr_ ? slab_mr_->rkey : 0; }

  bool is_using_host_memory() const { return using_host_memory_; }

  struct ibv_mr_info {
    ibv_mr *mr;
    size_t offset;
    uint32_t lkey;
    uint32_t rkey;
  };

  ibv_mr_info get_mr_info(const dptr &ptr) const {
    const void *ptr_addr = ptr.get<void>();
    size_t offset =
        static_cast<const uint8_t *>(ptr_addr) - static_cast<const uint8_t *>(slab_ptr_);

    return {slab_mr_, offset, slab_mr_->lkey, slab_mr_->rkey};
  }

private:
  std::multimap<size_t, size_t> free_blocks_; // size -> offset
  const Device &device_;
  ibv_pd *pd_;
  ibv_mr *slab_mr_;
  void *slab_ptr_;
  size_t slab_size_;
  size_t allocated_;
  bool using_host_memory_;
  mutable std::mutex mutex_;

  dptr create_dptr(size_t offset, size_t size) {
    auto *storage_info = new slab_storage_info{this, offset, size};

    void *slice_ptr = static_cast<uint8_t *>(slab_ptr_) + offset;
    auto storage = std::shared_ptr<device_storage>(
        new device_storage(&device_, slice_ptr, size, DEFAULT_ALIGNMENT),
        [storage_info](device_storage *storage) {
          // custom deleter to reclaim memory back to the allocator
          if (storage_info && storage_info->allocator) {
            storage_info->allocator->reclaim(storage_info->offset, storage_info->size);
          }
          delete storage_info;
          storage->ptr = nullptr; // Prevent device_storage destructor from freeing
          delete storage;
        });

    return dptr(storage, 0, size);
  }

  void reclaim(size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.emplace(size, offset);

#ifndef NDEBUG
    std::cout << "IbvAllocator: Reclaimed block of size " << size << " at offset " << offset
              << "\n";
#endif
  }

  struct slab_storage_info {
    IbvAllocator *allocator;
    size_t offset;
    size_t size;
  };
};

} // namespace tnn
