/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <infiniband/verbs.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"
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
      : device_(device),
        pd_(pd),
        slab_size_(slab_size),
        using_host_memory_(true) {
    if (!pd_) {
      throw std::invalid_argument("Protection Domain cannot be null");
    }
    if (slab_size_ == 0) {
      throw std::invalid_argument("Slab size must be greater than 0");
    }

    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    if (posix_memalign(&slab_ptr_, DEFAULT_ALIGNMENT, slab_size_) != 0) {
      throw std::runtime_error("Failed to allocate host pinned memory");
    }

    slab_mr_ = ibv_reg_mr(pd_, slab_ptr_, slab_size_, access_flags);
    if (!slab_mr_) {
      int err = errno;
      free(slab_ptr_);
      slab_ptr_ = nullptr;
      throw std::runtime_error(
          "Failed to register host memory with InfiniBand: " + std::string(std::strerror(err)) +
          " (errno=" + std::to_string(err) + ")");
    }

    free_blocks_by_size_.emplace(slab_size_, 0);
    free_blocks_by_offset_.emplace(0, slab_size_);

#ifndef NDEBUG
    std::cout << "IbvAllocator: Registered host pinned memory slab of " << slab_size_
              << " bytes at " << slab_ptr_ << " with lkey=" << slab_mr_->lkey << "\n";
#endif
  }

  ~IbvAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slab_mr_) {
      ibv_dereg_mr(slab_mr_);
      slab_mr_ = nullptr;
    }

    if (slab_ptr_) {
      if (using_host_memory_) {
        free(slab_ptr_);
      } else {
        device_.deallocateAlignedMemory(slab_ptr_);
      }
      slab_ptr_ = nullptr;
    }

    free_blocks_by_size_.clear();
    free_blocks_by_offset_.clear();
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

    // Align size to DEFAULT_ALIGNMENT
    size_t align = DEFAULT_ALIGNMENT;
    size = (size + align - 1) / align * align;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_by_size_.lower_bound(size);

    if (it != free_blocks_by_size_.end()) {
      size_t block_size = it->first;
      size_t offset = it->second;
      free_blocks_by_size_.erase(it);
      free_blocks_by_offset_.erase(offset);

      if (block_size > size) {
        size_t remaining_size = block_size - size;
        size_t remaining_offset = offset + size;
        free_blocks_by_offset_[remaining_offset] = remaining_size;
        free_blocks_by_size_.emplace(remaining_size, remaining_offset);
      }

      return create_dptr(offset, size);
    }

    throw std::runtime_error("IbvAllocator: Out of memory in master slab");
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_by_size_.clear();
    free_blocks_by_offset_.clear();
    free_blocks_by_size_.emplace(slab_size_, 0);
    free_blocks_by_offset_.emplace(0, slab_size_);
  }

  void reserve(size_t size) override {
    return;  // no op
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_by_size_.size();
  }

  size_t allocated_bytes() const {
    // slab_size_ is constant, cached_bytes() handles locking internally
    return slab_size_ - cached_bytes();
  }

  size_t cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_by_size_) {
      total += pair.first;
    }
    return total;
  }

  size_t slab_size() const { return slab_size_; }

  const Device &device() const override { return device_; }

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
    const void *ptr_addr = ptr.get();
    size_t offset =
        static_cast<const uint8_t *>(ptr_addr) - static_cast<const uint8_t *>(slab_ptr_);

    return {slab_mr_, offset, slab_mr_->lkey, slab_mr_->rkey};
  }

private:
  std::multimap<size_t, size_t> free_blocks_by_size_;  // size -> offset
  std::map<size_t, size_t> free_blocks_by_offset_;     // offset -> size
  const Device &device_;
  ibv_pd *pd_;
  ibv_mr *slab_mr_;
  void *slab_ptr_;
  size_t slab_size_;
  bool using_host_memory_;
  mutable std::mutex mutex_;

  dptr create_dptr(size_t offset, size_t size) {
    auto *storage_info = new slab_storage_info{this, offset, size};

    void *slice_ptr = static_cast<uint8_t *>(slab_ptr_) + offset;
    auto storage = std::shared_ptr<device_storage>(
        new device_storage(getHost(), slice_ptr, size, DEFAULT_ALIGNMENT),
        [storage_info](device_storage *storage) {
          // custom deleter to reclaim memory back to the allocator
          if (storage_info && storage_info->allocator) {
            storage_info->allocator->reclaim(storage_info->offset, storage_info->size);
          }
          delete storage_info;
          delete storage;
        });

    return dptr(storage, 0, size);
  }

  void remove_from_size_map(size_t size, size_t offset) {
    auto range = free_blocks_by_size_.equal_range(size);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == offset) {
        free_blocks_by_size_.erase(it);
        break;
      }
    }
  }

  void reclaim(size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_blocks_by_offset_.upper_bound(offset);

    // coalesce with next block
    if (it != free_blocks_by_offset_.end() && offset + size == it->first) {
      size += it->second;
      remove_from_size_map(it->second, it->first);
      it = free_blocks_by_offset_.erase(it);
    }

    // coalesce with previous block
    if (it != free_blocks_by_offset_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second == offset) {
        offset = prev->first;
        size += prev->second;
        remove_from_size_map(prev->second, prev->first);
        free_blocks_by_offset_.erase(prev);
      }
    }

    free_blocks_by_offset_[offset] = size;
    free_blocks_by_size_.emplace(size, offset);
  }

  struct slab_storage_info {
    IbvAllocator *allocator;
    size_t offset;
    size_t size;
  };
};

}  // namespace tnn
