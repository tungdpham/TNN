/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <set>
#ifndef NDEBUG
#include <fmt/core.h>

#include <iostream>
#endif

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

struct Slab {
public:
  Slab(void *ptr, size_t size)
      : ptr(ptr),
        size(size),
        left_offset(0),
        right_offset(size),
        active_allocations(0) {}

  ~Slab() {}

  Slab(const Slab &) = delete;
  Slab &operator=(const Slab &) = delete;

  Slab(Slab &&other) noexcept = default;

  Slab &operator=(Slab &&other) noexcept = default;

  void *ptr;
  size_t size;
  size_t left_offset;
  size_t right_offset;
  size_t active_allocations;
  std::map<size_t, size_t> free_by_offset;  // offset -> size

  void add_block(size_t offset, size_t size) {
    free_by_offset[offset] = size;
    // Add to free_by_size_ in parent allocator
  }

  void remove_block(size_t offset, size_t size) {
    free_by_offset.erase(offset);
    // Remove from free_by_size_ in parent allocator
  }

  size_t available_space() const { return right_offset - left_offset; }
};

// Double-ended List Allocator. Thread-safe, efficient workspace allocator that flips for
// input/output allocation patterns, with fallback to coalescing free-lists.
class DELAllocatorV2 : public IAllocator, public std::enable_shared_from_this<DELAllocatorV2> {
private:
  constexpr static size_t DEFAULT_SLAB_SIZE = 64 * 1024 * 1024;  // 64MB

  DELAllocatorV2(const Device &device, flowHandle_t flow)
      : device_(device),
        flow_(flow),
        side_(0) {}  // Added tracking counter

public:
  static std::shared_ptr<DELAllocatorV2> create(const Device &device, flowHandle_t flow) {
    return std::shared_ptr<DELAllocatorV2>(new DELAllocatorV2(device, flow));
  }

  static std::shared_ptr<DELAllocatorV2> instance(const Device &device, flowHandle_t flow) {
    static std::mutex mutex;
    static std::map<std::pair<DeviceType, flowHandle_t>, std::shared_ptr<DELAllocatorV2>> instances;

    std::lock_guard<std::mutex> lock(mutex);

    auto key = std::make_pair(device.device_type(), flow);
    auto it = instances.find(key);
    if (it != instances.end()) {
      return it->second;
    }
    auto instance = create(device, flow);
    instances[key] = instance;
    return instance;
  }

  ~DELAllocatorV2() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &slab : slabs_) {
      device_.deallocateAlignedMemory(slab.ptr);
    }
    slabs_.clear();
    free_by_size_.clear();
  }

  DELAllocatorV2(const DELAllocatorV2 &) = delete;
  DELAllocatorV2 &operator=(const DELAllocatorV2 &) = delete;

  dptr allocate(size_t size) override {
    if (size == 0) return dptr(nullptr);
    std::lock_guard<std::mutex> lock(mutex_);
    size_t aligned_size = align_up(size, DEFAULT_ALIGNMENT);
    // try to allocate from slab's middle region first
    for (auto it = slabs_.begin(); it != slabs_.end(); ++it) {
      Slab &slab = *it;
      if (slab.available_space() >= aligned_size) {
        size_t offset = (side_ == 0) ? slab.left_offset : (slab.right_offset - aligned_size);
        if (side_ == 0) {
          slab.left_offset += aligned_size;
        } else {
          slab.right_offset -= aligned_size;
        }
        return create_dptr(&slab, offset, size, aligned_size);
      }
    }

    // if no in between space, try to allocate from free blocks
    auto it = free_by_size_.lower_bound(aligned_size);
    while (it != free_by_size_.end()) {
      if (it->second.empty()) {
        // Clean up orphaned empty buckets left behind by previous operations
        it = free_by_size_.erase(it);
        continue;
      }

      std::set<Block> &blocks = it->second;
      Block block = *blocks.begin();
      blocks.erase(blocks.begin());

      if (blocks.empty()) {
        free_by_size_.erase(it);
      }

      block.slab->remove_block(block.offset, block.size);

      // handle remainder
      size_t remainder = block.size - aligned_size;
      if (remainder > 0) {
        size_t new_offset = block.offset + aligned_size;
        block.slab->add_block(new_offset, remainder);
        free_by_size_[remainder].insert({block.slab, new_offset, remainder});
      }

      return create_dptr(block.slab, block.offset, size, aligned_size);
    }

    // if still no space, allocate a new slab
    size_t slab_size = aligned_size;
    Slab &slab = allocate_slab(slab_size);
    size_t offset = (side_ == 0) ? slab.left_offset : (slab.right_offset - aligned_size);
    if (side_ == 0) {
      slab.left_offset += aligned_size;
    } else {
      slab.right_offset -= aligned_size;
    }
    return create_dptr(&slab, offset, size, aligned_size);
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
    for (auto &slab : slabs_) {
      if (slab.active_allocations > 0) {
        throw std::runtime_error("DELAllocatorV2: Cannot clear with active allocations");
      }
      device_.deallocateAlignedMemory(slab.ptr);
    }
    slabs_.clear();
    free_by_size_.clear();
  }

  void reserve(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t aligned_size = align_up(size, DEFAULT_ALIGNMENT);

    for (const auto &slab : slabs_) {
      if (slab.available_space() >= aligned_size) {
        return;  // already have enough free space
      }
    }

    auto it = free_by_size_.lower_bound(aligned_size);
    if (it != free_by_size_.end()) {
      return;  // already have enough in free blocks
    }

    allocate_slab(aligned_size);
  }

  const Device &device() const override { return device_; }

private:
  struct Block {
    Slab *slab;
    size_t offset;
    size_t size;

    bool operator<(const Block &other) const {
      if (slab != other.slab) return slab < other.slab;
      return offset < other.offset;
    }
  };
  const Device &device_;
  flowHandle_t flow_;
  std::mutex mutex_;
  int side_;
  std::list<Slab> slabs_;
  std::map<size_t, std::set<Block>> free_by_size_;  // size -> set of blocks

  dptr create_dptr(Slab *slab, size_t offset, size_t size, size_t actual_size) {
    void *slice_ptr = static_cast<unsigned char *>(slab->ptr) + offset;

    slab->active_allocations++;

    auto self_shared = shared_from_this();

    auto storage = std::shared_ptr<device_storage>(
        new device_storage(device_, slice_ptr, size, DEFAULT_ALIGNMENT),
        [self_shared, slab, offset, actual_size](device_storage *storage) {
          self_shared->reclaim(slab, offset, actual_size);
          delete storage;
        });

    return dptr(storage, 0, size);
  }

  void reclaim(Slab *slab, size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    slab->active_allocations--;

    // coalesce forward
    auto next_it = slab->free_by_offset.lower_bound(offset + size);
    if (next_it != slab->free_by_offset.end() && offset + size == next_it->first) {
      size_t next_size = next_it->second;
      size += next_size;

      free_by_size_[next_size].erase({slab, next_it->first, next_size});
      if (free_by_size_[next_size].empty()) free_by_size_.erase(next_size);
      slab->free_by_offset.erase(next_it);
    }

    // coalesce backward
    auto prev_it = slab->free_by_offset.lower_bound(offset);
    if (prev_it != slab->free_by_offset.begin()) {
      --prev_it;
      if (prev_it->first + prev_it->second == offset) {
        size_t prev_size = prev_it->second;
        offset = prev_it->first;
        size += prev_size;

        free_by_size_[prev_size].erase({slab, prev_it->first, prev_size});
        if (free_by_size_[prev_size].empty()) free_by_size_.erase(prev_size);
        slab->free_by_offset.erase(prev_it);
      }
    }

    // check if we can just roll back the bump allocator offsets
    if (offset + size == slab->left_offset) {
      slab->left_offset = offset;
    } else if (offset == slab->right_offset) {
      slab->right_offset = offset + size;
    } else {
      // Add coalesced block to free lists
      slab->free_by_offset[offset] = size;
      free_by_size_[size].insert({slab, offset, size});
    }

    if (slab->active_allocations == 0) {
      merge_slabs();
    }
  }

  void merge_slabs() {
    size_t total_free_size = 0;
    int empty_slab_count = 0;
    for (auto &slab : slabs_) {
      if (slab.active_allocations == 0) {
        total_free_size += slab.size;
        empty_slab_count++;
      }
    }

    if (empty_slab_count == 1) {
      // only one empty slab, no need to merge
      return;
    }

    // merge all empty slabs into one
    for (auto it = slabs_.begin(); it != slabs_.end();) {
      if (it->active_allocations == 0) {
        for (const auto &[block_offset, block_size] : it->free_by_offset) {
          free_by_size_[block_size].erase({&*it, block_offset, block_size});
          if (free_by_size_[block_size].empty()) free_by_size_.erase(block_size);
        }
        device_.deallocateAlignedMemory(it->ptr);
        it = slabs_.erase(it);
      } else {
        ++it;
      }
    }

    // allocate the new slab from the combined total size
#ifndef NDEBUG
    std::cout << fmt::format("DELAllocatorV2: Merging {} empty slabs into a new {} byte slab.",
                             empty_slab_count, total_free_size)
              << std::endl;
#endif
    allocate_slab(total_free_size);
  }

  Slab &allocate_slab(size_t slab_size) {
    size_t freed_bytes = 0;
    // free any existing empty slabs to maximize available memory before allocating a new one
    for (auto it = slabs_.begin(); it != slabs_.end();) {
      if (it->active_allocations == 0) {
        assert(it->free_by_offset.empty() && "Empty slab should have no free blocks");
        device_.deallocateAlignedMemory(it->ptr);
        freed_bytes += it->size;
        it = slabs_.erase(it);
      } else {
        ++it;
      }
    }
    slab_size = std::max(slab_size, freed_bytes);
    slab_size = align_up(slab_size, DEFAULT_ALIGNMENT);
    void *slab_ptr = device_.allocateAlignedMemory(slab_size, DEFAULT_ALIGNMENT);
    if (!slab_ptr) {
      throw std::runtime_error("DELAllocatorV2: Failed to allocate slab");
    }
    Slab &slab = slabs_.emplace_back(slab_ptr, slab_size);
    // do not add to free_by_size_, let bump allocation use left/right offsets.

#ifndef NDEBUG
    std::cout << fmt::format("DELAllocatorV2: Allocated new slab of size {} bytes", slab_size)
              << std::endl;
#endif
    return slab;
  }

  void free_slab(Slab *slab) {
    if (slab->active_allocations > 0) {
      throw std::runtime_error("Cannot free slab with active allocations");
    }
    device_.deallocateAlignedMemory(slab->ptr);
    slabs_.remove_if([slab](const Slab &s) { return &s == slab; });
  }
};

}  // namespace tnn