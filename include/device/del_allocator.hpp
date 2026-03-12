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
#include <stdexcept>

#include "device/dptr.hpp"
#include "device/flow.hpp"
#include "device/iallocator.hpp"

namespace tnn {

inline size_t align_up(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

inline size_t align_down(size_t size, size_t alignment) { return size & ~(alignment - 1); }

// Double-ended List Allocator. Thread-safe, efficient workspace allocator that flips for
// input/output allocation patterns.
class DELAllocator : public IAllocator, public std::enable_shared_from_this<DELAllocator> {
private:
  DELAllocator(const Device &device, flowHandle_t flow)
      : device_(device),
        flow_(flow),
        slab_ptr_(nullptr),
        left_offset_(0),
        right_offset_(0),
        side_(0) {}

public:
  static std::shared_ptr<DELAllocator> create(const Device &device, flowHandle_t flow) {
    return std::shared_ptr<DELAllocator>(new DELAllocator(device, flow));
  }

  // should be kept alive longer than any allocated dptrs
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

    if (side_ == 0) {
      left_offset_ = align_up(left_offset_, DEFAULT_ALIGNMENT);
      size = align_up(size, DEFAULT_ALIGNMENT);
      if (left_offset_ + size > right_offset_) {
        throw std::runtime_error("DELAllocator: Out of memory");
      }
      size_t offset = left_offset_;
      left_offset_ += size;
      side_ = 1 - side_;  // flip
      return create_dptr(offset, size);
    } else {
      size = align_up(size, DEFAULT_ALIGNMENT);
      right_offset_ = align_down(right_offset_, DEFAULT_ALIGNMENT);
      if (right_offset_ < left_offset_ + size) {
        throw std::runtime_error("DELAllocator: Out of memory");
      }
      right_offset_ -= size;
      size_t offset = right_offset_;
      side_ = 1 - side_;  // flip
      return create_dptr(offset, size);
    }
  }

  void flip() {
    std::lock_guard<std::mutex> lock(mutex_);
    side_ = 1 - side_;
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    left_offset_ = 0;
    right_offset_ = capacity_;
  }

  void reserve(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (capacity_ >= size) {
      return;  // already have enough capacity
    }
    if (left_offset_ != 0 || right_offset_ != capacity_) {
      throw std::runtime_error(
          "DELAllocator: Cannot reserve while there are outstanding allocations");
    }
    capacity_ = align_up(size, DEFAULT_ALIGNMENT);
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
  std::map<size_t, size_t> allocated_blocks_;  // offset -> size
  int side_;

  dptr create_dptr(size_t offset, size_t size) {
    void *slice_ptr = static_cast<uint8_t *>(slab_ptr_) + offset;

    allocated_blocks_.insert({offset, size});

    std::weak_ptr<DELAllocator> self_weak = shared_from_this();

    auto storage = std::shared_ptr<device_storage>(
        new device_storage(device_, slice_ptr, size, DEFAULT_ALIGNMENT),
        [self_weak, offset, size](device_storage *storage) {
          if (auto self = self_weak.lock()) {
            self->reclaim(offset, size);
          }
          delete storage;
        });

    return dptr(storage, 0, size);
  }

  void reclaim(size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_blocks_.find(offset);
    if (it != allocated_blocks_.end()) {
      allocated_blocks_.erase(it);
    }
    // shift left or right offset based on which side this block is on
    if (offset < left_offset_) {
      auto it = allocated_blocks_.lower_bound(left_offset_);
      if (it != allocated_blocks_.begin()) {
        --it;
        left_offset_ = it->first + it->second;
      } else {
        left_offset_ = 0;
      }
    } else {
      auto it = allocated_blocks_.lower_bound(right_offset_);
      if (it != allocated_blocks_.end()) {
        right_offset_ = it->first;
      } else {
        right_offset_ = capacity_;
      }
    }
  }
};
}  // namespace tnn
