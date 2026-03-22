/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <deque>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"

namespace tnn {
// the buffer of all time. almost like linked list of buffers
class VBuffer {
private:
  constexpr static size_t MAX_K_INLINE_BUFFERS = 10;  // max number of inline buffers
  IAllocator &allocator_;
  std::deque<dptr> buffers_;  // inline buffer list
  size_t size_ = 0;

  void ensure(size_t required_size) {
    if (capacity() < required_size) {
      throw std::runtime_error("VBuffer does not have enough capacity to ensure required size");
    }
    size_ = required_size;
  }

  void check_offset(size_t required_size) const {
    if (required_size > size_) {
      throw std::out_of_range("Offset exceeds the size of VBuffer");
    }
  }

public:
  VBuffer(IAllocator &allocator)
      : allocator_(allocator) {}

  VBuffer(const VBuffer &) = delete;
  VBuffer &operator=(const VBuffer &) = delete;
  VBuffer(VBuffer &&) noexcept = default;

  size_t size() const { return size_; }

  size_t capacity() const {
    size_t total_capacity = 0;
    for (const auto &buf : buffers_) {
      total_capacity += buf.capacity();
    }
    return total_capacity;
  }

  void resize(size_t new_size) {
    if (new_size > capacity()) {
      throw std::runtime_error("Cannot resize VBuffer to a size larger than its capacity");
    }
    size_ = new_size;
  }

  void reset() { size_ = 0; }

  void clear() {
    buffers_.clear();
    size_ = 0;
  }

  bool empty() const { return buffers_.empty(); }

  // allocate a new buffer
  void alloc(size_t size) {
    dptr new_buffer = allocator_.allocate(size);  // new buffer may be larger than requested
    buffers_.emplace_back(new_buffer.span(0, size));
  }

  dptr poll() {
    if (buffers_.empty()) {
      throw std::runtime_error("VBuffer is empty, cannot poll");
    }
    dptr front = std::move(buffers_.front());
    buffers_.pop_front();
    if (size_ > front.capacity()) {
      size_ -= front.capacity();
    } else {
      size_ = 0;
    }
    return front;
  }

  // returns dptr to the data at offset
  dptr get(size_t offset) {
    if (buffers_.empty()) {
      throw std::runtime_error("Failed to get from empty VBuffer");
    }
    if (offset > size_) {
      throw std::out_of_range("Offset is out of range");
    }
    size_t current_offset = 0;
    auto it = buffers_.begin();
    while (it != buffers_.end() && current_offset + it->capacity() <= offset) {
      current_offset += it->capacity();
      ++it;
    }
    if (it == buffers_.end()) {
      return dptr(nullptr);
    }
    return it->span(offset - current_offset, it->capacity() - (offset - current_offset));
  }

  const dptr get(size_t offset) const { return const_cast<VBuffer *>(this)->get(offset); }
};
}  // namespace tnn