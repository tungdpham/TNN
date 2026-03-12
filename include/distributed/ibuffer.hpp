/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "endian.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {

class IBuffer {
protected:
  dptr data_;
  size_t size_;
  // Endianess of the data in the buffer, does not apply to write. Read functions will swap bytes if
  // needed.
  mutable Endianness endianess_ =
      get_system_endianness();  // 0 for little-endian and 1 for big-endian

  static constexpr size_t MAX_ARRAY_SIZE = static_cast<size_t>(-1) - 8;

  bool in_range(size_t index) const { return index < size_; }

  std::string get_out_of_bound_msg(int index) const {
    return "Array bounds is (0, " + std::to_string(size_) + "), accessed: " + std::to_string(index);
  }

public:
  IBuffer()
      : data_(nullptr),
        size_(0) {}

  IBuffer(IAllocator &allocator, size_t capacity)
      : data_(allocator.allocate(capacity)),
        size_(0) {}

  IBuffer(dptr &&data)
      : data_(std::move(data)),
        size_(0) {}

  IBuffer(const IBuffer &other)
      : data_(other.data_),
        size_(other.size_) {}

  IBuffer(IBuffer &&other) noexcept
      : data_(std::move(other.data_)),
        size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  ~IBuffer() = default;

  IBuffer &operator=(const IBuffer &other) = delete;

  IBuffer &operator=(IBuffer &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      size_ = other.size_;
      other.size_ = 0;
    }
    return *this;
  }

  void set_endianess(Endianness endianess) const { endianess_ = endianess; }

  Endianness get_endianess() const { return endianess_; }

  void *data() { return data_.get<void>(); }
  const void *data() const { return data_.get<void>(); }

  dptr span(size_t offset, size_t length) {
    if (offset + length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + length));
    }
    return data_.span(offset, length);
  }

  const dptr span(size_t offset, size_t length) const {
    if (offset + length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + length));
    }
    return data_.span(offset, length);
  }

  bool empty() const { return size_ == 0; }
  size_t size() const { return size_; }
  size_t capacity() const { return data_.capacity(); }
  void reset() { size_ = 0; }

#if defined(ARCH_64)
  template <typename T>
  inline void write(size_t &offset, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure_capacity(offset + sizeof(T));
    std::memcpy(data_.get<uchar>() + offset, &value, sizeof(T));
    if (offset + sizeof(T) > size_) {
      size_ = offset + sizeof(T);
    }
    offset += sizeof(T);
  }

  template <typename T>
  inline void write(size_t &offset, const T *arr, size_t length) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    ensure_capacity(offset + byte_size);
    std::memcpy(data_.get<uchar>() + offset, arr, byte_size);
    if (offset + length * sizeof(T) > size_) {
      size_ = offset + length * sizeof(T);
    }
    offset += length * sizeof(T);
  }

  inline void write(size_t &offset, const dptr &ptr, size_t length) {
    ensure_capacity(offset + length);
    ptr.copy_to_host(data_.get<uchar>() + offset, length);
    if (offset + length > size_) {
      size_ = offset + length;
    }
    offset += length;
  }

  inline void write(size_t &offset, const std::string &str) {
    uint64_t str_length = static_cast<uint64_t>(str.size());
    write<uint64_t>(offset, str_length);
    if (str_length > 0) {
      write(offset, str.data(), str_length);
    }
  }

  template <typename T>
  inline void read(size_t &offset, T &value) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    if (offset + sizeof(T) > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + sizeof(T)));
    }
    std::memcpy(&value, data_.get<uchar>() + offset, sizeof(T));
    offset += sizeof(T);
    if (endianess_ != get_system_endianness()) {
      bswap(value);
    }
  }

  template <typename T>
  inline void read(size_t &offset, T *arr, size_t length) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    if (offset + byte_size > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + byte_size));
    }
    std::memcpy(arr, data_.get<uchar>() + offset, byte_size);
    // should not happen often
    if (endianess_ != get_system_endianness()) {
      parallel_for<size_t>(0, length, [&](size_t i) { bswap(arr[i]); });
    }
    offset += byte_size;
  }

  inline void read(size_t &offset, dptr &ptr, size_t length) const {
    if (offset + length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + length));
    }
    ptr.copy_from_host(data_.get<uchar>() + offset, length);
    offset += length;
  }

  inline void read(size_t &offset, std::string &str) const {
    uint64_t str_length;
    read<uint64_t>(offset, str_length);
    if (offset + str_length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + str_length));
    }
    if (str_length > 0) {
      str.resize(str_length);
      read(offset, str.data(), str_length);
    } else {
      str.clear();
    }
  }
#elif defined(ARCH_32)
  // Implementations for 32-bit architectures can go here

#else
#error "Unknown architecture. Codebase supports only 32-bit and 64-bit architectures."
#endif

  void resize(size_t new_size) {
    if (new_size > data_.capacity()) {
      throw std::runtime_error("IBuffer: resize exceeds buffer capacity");
    }
    size_ = new_size;
  }

private:
  void ensure_capacity(size_t required_capacity) {
    if (required_capacity > data_.capacity()) {
      throw std::runtime_error("IBuffer: write exceeds buffer capacity");
    }
  }
};

}  // namespace tnn
