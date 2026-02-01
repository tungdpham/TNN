/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <deque>

#include "device/dptr.hpp"

namespace tnn {
// the buffer of all time. almost like linked list of buffers
class VBuffer {
private:
  std::deque<dptr> buffers_;  // inline buffer list
  size_t size_ = 0;

  void check_offset(size_t offset) const {
    if (buffers_.empty()) {
      throw std::runtime_error("Failed to get from empty VBuffer");
    }
    if (offset > size_) {
      throw std::out_of_range("Offset is out of range");
    }
  }

public:
  VBuffer() = default;

  VBuffer(const VBuffer &) = delete;
  VBuffer &operator=(const VBuffer &) = delete;
  VBuffer(VBuffer &&) noexcept = default;
  VBuffer &operator=(VBuffer &&) noexcept = default;

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
  void alloc(dptr &&buffer) { buffers_.emplace_back(std::move(buffer)); }

  dptr poll() {
    if (buffers_.empty()) {
      throw std::runtime_error("VBuffer is empty, cannot poll");
    }
    dptr front = std::move(buffers_.front());
    buffers_.pop_front();
    size_ -= front.capacity();
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
    while (it != buffers_.end() && current_offset + it->capacity() < offset) {
      current_offset += it->capacity();
      ++it;
    }
    return it->span(offset - current_offset, it->capacity() - (offset - current_offset));
  }

  const dptr get(size_t offset) const { return const_cast<VBuffer *>(this)->get(offset); }

  template <typename T>
  inline void write(size_t &offset, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    check_offset(offset + sizeof(T));
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
    check_offset(offset + sizeof(T) * length);
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
    check_offset(offset + str_length);
    if (str_length > 0) {
      str.resize(str_length);
      read(offset, str.data(), str_length);
    } else {
      str.clear();
    }
  }
};
}  // namespace tnn