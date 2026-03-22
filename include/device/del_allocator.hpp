/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifndef NDEBUG
#include <fmt/core.h>

#include <iostream>
#endif

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>

#include "device/dptr.hpp"
#include "device/flow.hpp"
#include "device/iallocator.hpp"

namespace tnn {

inline size_t align_up(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Double-ended List Allocator. Thread-safe, efficient workspace allocator that flips for
// input/output allocation patterns, with fallback to coalescing free-lists.
class DELAllocator : public IAllocator, public std::enable_shared_from_this<DELAllocator> {
private:
  DELAllocator(const Device &device, flowHandle_t flow)
      : device_(device),
        flow_(flow),
        slab_ptr_(nullptr),
        capacity_(0),
        left_offset_(0),
        right_offset_(0),
        active_allocations_(0),
        side_(0) {}  // Added tracking counter

public:
  static std::shared_ptr<DELAllocator> create(const Device &device, flowHandle_t flow) {
    return std::shared_ptr<DELAllocator>(new DELAllocator(device, flow));
  }

  ~DELAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slab_ptr_) {
      device_.deallocateAlignedMemory(slab_ptr_);
      slab_ptr_ = nullptr;
    }
  }

  DELAllocator(const DELAllocator &) = delete;
  DELAllocator &operator=(const DELAllocator &) = delete;

  dptr allocate(size_t size) override {
    if (size == 0) return dptr(nullptr);
    std::lock_guard<std::mutex> lock(mutex_);

    size_t aligned_size = align_up(size, DEFAULT_ALIGNMENT);
    bool fallback = false;
    size_t offset = 0;

    if (side_ == 0) {
      if (left_offset_ + aligned_size <= right_offset_) {
        offset = left_offset_;
        left_offset_ += aligned_size;
      } else {
        fallback = true;
      }
    } else {
      if (right_offset_ >= left_offset_ + aligned_size) {
        right_offset_ -= aligned_size;
        offset = right_offset_;
      } else {
        fallback = true;
      }
    }

    if (!fallback) {
      return create_dptr(offset, aligned_size);
    }

    auto it = free_by_size_.lower_bound(aligned_size);
    if (it != free_by_size_.end()) {
      size_t block_size = it->first;
      size_t block_offset = *it->second.begin();

      remove_block(block_offset, block_size);

      size_t remainder = block_size - aligned_size;
      if (remainder > 0) {
        add_block(block_offset + aligned_size, remainder);
      }

      return create_dptr(block_offset, aligned_size);
    }

    throw std::runtime_error("DELAllocator: Out of memory");
  }

  void flip() {
    std::lock_guard<std::mutex> lock(mutex_);
    side_ = 1 - side_;
  }

  void set_side(int side) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (side != 0 && side != 1) {
      throw std::invalid_argument("DELAllocator: Side must be 0 or 1");
    }
    side_ = side;
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_allocations_ > 0) {
      throw std::runtime_error("DELAllocator: Cannot clear while allocations are active");
    }

    left_offset_ = 0;
    right_offset_ = capacity_;
    free_by_offset_.clear();
    free_by_size_.clear();
  }

  void reserve(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (capacity_ >= size) {
      return;  // already have enough capacity
    }

    if (active_allocations_ > 0 || left_offset_ != 0 || right_offset_ != capacity_) {
      throw std::runtime_error(
          "DELAllocator: Cannot reserve while there are outstanding allocations");
    }

    capacity_ = align_up(size, DEFAULT_ALIGNMENT);

    if (slab_ptr_) {
      device_.deallocateAlignedMemory(slab_ptr_);
    }

    slab_ptr_ = device_.allocateAlignedMemory(capacity_, DEFAULT_ALIGNMENT);
    if (!slab_ptr_) {
      throw std::runtime_error("DELAllocator: Failed to allocate master slab of size " +
                               std::to_string(capacity_) + " bytes");
    }
    right_offset_ = capacity_;
#ifndef NDEBUG
    std::cout << fmt::format("DELAllocator: Reserved master slab of size {} bytes", capacity_)
              << std::endl;
#endif
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return 1;
  }

  size_t cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  const Device &device() const override { return device_; }

private:
  const Device &device_;
  flowHandle_t flow_;
  mutable std::mutex mutex_;
  void *slab_ptr_;
  size_t capacity_;
  size_t left_offset_;
  size_t right_offset_;
  size_t active_allocations_;

  std::map<size_t, size_t> free_by_offset_;          // offset -> size
  std::map<size_t, std::set<size_t>> free_by_size_;  // size -> set of offsets

  int side_;

  void add_block(size_t offset, size_t size) {
    free_by_offset_[offset] = size;
    free_by_size_[size].insert(offset);
  }

  void remove_block(size_t offset, size_t size) {
    free_by_offset_.erase(offset);
    auto it = free_by_size_.find(size);
    if (it != free_by_size_.end()) {
      it->second.erase(offset);
      if (it->second.empty()) {
        free_by_size_.erase(it);
      }
    }
  }

  dptr create_dptr(size_t offset, size_t size) {
    void *slice_ptr = static_cast<uint8_t *>(slab_ptr_) + offset;

    active_allocations_++;

    auto self_shared = shared_from_this();

    auto storage = std::shared_ptr<device_storage>(
        new device_storage(device_, slice_ptr, size, DEFAULT_ALIGNMENT),
        [self_shared, offset, size](device_storage *storage) {
          self_shared->reclaim(offset, size);
          delete storage;
        });

    return dptr(storage, 0, size);
  }

  void reclaim(size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (active_allocations_ > 0) {
      active_allocations_--;
    }

    auto next_it = free_by_offset_.lower_bound(offset + size);
    if (next_it != free_by_offset_.end() && offset + size == next_it->first) {
      size += next_it->second;
      remove_block(next_it->first, next_it->second);
    }

    auto prev_it = free_by_offset_.lower_bound(offset);
    if (prev_it != free_by_offset_.begin()) {
      --prev_it;
      if (prev_it->first + prev_it->second == offset) {
        offset = prev_it->first;
        size += prev_it->second;
        remove_block(prev_it->first, prev_it->second);
      }
    }

    if (offset + size == left_offset_) {
      left_offset_ = offset;
    } else if (offset == right_offset_) {
      right_offset_ = offset + size;
    } else {
      add_block(offset, size);
    }
  }
};

}  // namespace tnn