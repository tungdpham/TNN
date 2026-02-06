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
  constexpr static size_t MAX_K_INLINE_BUFFERS = 4;  // max number of inline buffers
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

  template <typename T>
  inline void write(size_t &offset, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure(offset + sizeof(T));
    dptr ptr = get(offset);
    if (ptr.capacity() < sizeof(T)) {
      throw std::runtime_error("Write operation spans multiple buffers, which is not supported");
    }
    ptr.copy_from_host(&value, sizeof(T));
    offset += sizeof(T);
  }

  template <typename T>
  inline void write(size_t &offset, const T *arr, size_t length) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure(offset + sizeof(T) * length);
    size_t byte_size = sizeof(T) * length;
    dptr ptr = get(offset);
    if (ptr.capacity() < byte_size) {
      throw std::runtime_error(
          "Write operation spans multiple buffers, which is not supported yet");
    }
    ptr.copy_from_host(arr, byte_size);
    offset += byte_size;
  }

  inline void write(size_t &offset, std::string str) {
    write(offset, static_cast<uint64_t>(str.size()));
    if (!str.empty()) {
      write(offset, str.data(), str.size());
    }
  }

  // append an existing buffer to the end
  inline void append(dptr &&src) {
    // shrink capacity of last buffer to fit current size if needed
    if (!buffers_.empty() && size_ < capacity()) {
      size_t excess_capacity = capacity() - size_;
      while (excess_capacity > buffers_.back().capacity()) {
        excess_capacity -= buffers_.back().capacity();
        buffers_.pop_back();
      }
      auto &last_buf = buffers_.back();
      last_buf = last_buf.span(0, last_buf.capacity() - excess_capacity);
    }
    size_ += src.capacity();
    buffers_.emplace_back(std::move(src));
  }

  template <typename T>
  inline void read(size_t &offset, T &value) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    check_offset(offset + sizeof(T));
    const dptr ptr = get(offset);
    if (ptr.capacity() < sizeof(T)) {
      throw std::runtime_error("Read operation spans multiple buffers, which is not supported");
    }
    ptr.copy_to_host(&value, sizeof(T));
    offset += sizeof(T);
  }

  template <typename T>
  inline void read(size_t &offset, T *arr, size_t length) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    check_offset(offset + sizeof(T) * length);
    size_t byte_size = sizeof(T) * length;
    const dptr ptr = get(offset);
    if (ptr.capacity() < byte_size) {
      throw std::runtime_error("Read operation spans multiple buffers, which is not supported yet");
    }
    ptr.copy_to_host(arr, byte_size);
    offset += byte_size;
  }

  inline void read(size_t &offset, std::string &str) const {
    uint64_t str_length;
    read(offset, str_length);
    if (str_length > 0) {
      str.resize(str_length);
      read(offset, str.data(), str_length);
    } else {
      str.clear();
    }
  }
};
}  // namespace tnn