/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/dptr.hpp"
#include "endian.hpp"
#include "ops/ops.hpp"
#include "threading/thread_handler.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <infiniband/verbs.h>
#include <stdexcept>
#include <string>

namespace tnn {

class RoceBuffer {
private:
  dptr data_;
  size_t size_;
  ibv_mr *mr_;
  Endianness endianess_ = get_system_endianness();

  static constexpr size_t MAX_ARRAY_SIZE = static_cast<size_t>(-1) - 8;
  static constexpr size_t DEFAULT_CAPACITY = 10;

  bool in_range(size_t index) const { return index < size_; }

  std::string get_out_of_bound_msg(size_t index) const {
    return "Array bounds is (0, " + std::to_string(size_) + "), accessed: " + std::to_string(index);
  }

  static size_t huge_capacity(size_t min_capacity) {
    if (min_capacity > MAX_ARRAY_SIZE) {
      throw std::bad_alloc();
    }
    return min_capacity > MAX_ARRAY_SIZE ? static_cast<size_t>(-1) : MAX_ARRAY_SIZE;
  }

public:
  RoceBuffer(ibv_mr *mr = nullptr, dptr data = nullptr, size_t initial_size = 0)
      : data_(data), size_(initial_size), mr_(mr) {
    if (initial_size > MAX_ARRAY_SIZE) {
      throw std::invalid_argument("Illegal Size: " + std::to_string(initial_size));
    }
    if (initial_size > 0 && !data_) {
      throw std::invalid_argument("Data pointer cannot be null with non-zero initial size");
    }
  }

  ~RoceBuffer() = default;

  RoceBuffer(const RoceBuffer &) = delete;
  RoceBuffer &operator=(const RoceBuffer &) = delete;

  RoceBuffer(RoceBuffer &&other) noexcept
      : data_(std::move(other.data_)), size_(other.size_), mr_(other.mr_) {
    other.size_ = 0;
    other.mr_ = nullptr;
  }

  RoceBuffer &operator=(RoceBuffer &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      size_ = other.size_;
      mr_ = other.mr_;

      other.size_ = 0;
      other.mr_ = nullptr;
    }
    return *this;
  }

  uint8_t &operator[](size_t index) {
    if (index >= size_) {
      throw std::out_of_range(get_out_of_bound_msg(index));
    }
    return data_.get<uint8_t>()[index];
  }

  const uint8_t &operator[](size_t index) const {
    if (index >= size_) {
      throw std::out_of_range(get_out_of_bound_msg(index));
    }
    return data_.get<const uint8_t>()[index];
  }

  void set_endianess(Endianness endianess) { endianess_ = endianess; }
  Endianness get_endianess() const { return endianess_; }

  uint8_t *get() { return data_.get<uint8_t>(); }
  const uint8_t *get() const { return data_.get<const uint8_t>(); }

  dptr &get_dptr() { return data_; }
  const dptr &get_dptr() const { return data_; }

  bool empty() const { return size_ == 0 || !data_; }
  size_t size() const { return size_; }
  size_t capacity() const { return data_ ? data_.capacity() : 0; }

  ibv_mr *get_mr() const { return mr_; }
  uint32_t get_lkey() const { return mr_ ? mr_->lkey : 0; }

  void set_mr(ibv_mr *mr) { mr_ = mr; }

  void reserve(size_t new_capacity) {
    if (new_capacity > capacity()) {
      throw std::runtime_error(
          "Cannot dynamically resize RoceBuffer with dptr. Pre-allocate sufficient capacity.");
    }
  }

  void clear() { size_ = 0; }

  template <typename T> inline void write(size_t &offset, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    if (offset + sizeof(T) > capacity()) {
      throw std::out_of_range("Write exceeds buffer capacity");
    }
    if (!data_) {
      throw std::runtime_error("Cannot write to null buffer");
    }
    std::memcpy(data_.get<uint8_t>() + offset, &value, sizeof(T));
    if (offset + sizeof(T) > size_) {
      size_ = offset + sizeof(T);
    }
    offset += sizeof(T);
  }

  template <typename T> inline void write(size_t &offset, const T *arr, size_t length) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    if (offset + byte_size > capacity()) {
      throw std::out_of_range("Write exceeds buffer capacity");
    }
    if (!data_) {
      throw std::runtime_error("Cannot write to null buffer");
    }
    std::memcpy(data_.get<uint8_t>() + offset, arr, byte_size);
    if (offset + byte_size > size_) {
      size_ = offset + byte_size;
    }
    offset += byte_size;
  }

  inline void write(size_t &offset, const dptr &ptr, size_t length) {
    if (offset + length > capacity()) {
      throw std::out_of_range("Write exceeds buffer capacity");
    }
    if (!data_) {
      throw std::runtime_error("Cannot write to null buffer");
    }
    ops::cd_copy<uint8_t>(ptr, data_, length);
    if (offset + length > size_) {
      size_ = offset + length;
    }
    offset += length;
  }

  inline void write(size_t &offset, const std::string &str) {
    uint64_t str_length = static_cast<uint64_t>(str.size());
    write<uint64_t>(offset, str_length);
    if (str_length > 0) {
      write(offset, reinterpret_cast<const uint8_t *>(str.data()), str_length);
    }
  }

  template <typename T> inline void read(size_t &offset, T &value) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    if (offset + sizeof(T) > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + sizeof(T)));
    }
    if (!data_) {
      throw std::runtime_error("Cannot read from null buffer");
    }
    std::memcpy(&value, data_.get<const uint8_t>() + offset, sizeof(T));
    offset += sizeof(T);
    if (endianess_ != get_system_endianness()) {
      bswap(value);
    }
  }

  template <typename T> inline void read(size_t &offset, T *arr, size_t length) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    if (offset + byte_size > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + byte_size));
    }
    if (!data_) {
      throw std::runtime_error("Cannot read from null buffer");
    }
    std::memcpy(arr, data_.get<const uint8_t>() + offset, byte_size);
    if (endianess_ != get_system_endianness()) {
      parallel_for<size_t>(0, length, [&](size_t i) { bswap(arr[i]); });
    }
    offset += byte_size;
  }

  inline void read(size_t &offset, dptr &ptr, size_t length) const {
    if (offset + length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + length));
    }
    if (!data_) {
      throw std::runtime_error("Cannot read from null buffer");
    }
    ops::cd_copy<uint8_t>(data_ + offset, ptr, length);
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
      read(offset, reinterpret_cast<uint8_t *>(str.data()), str_length);
    } else {
      str.clear();
    }
  }

  void resize(size_t new_size) {
    if (new_size > capacity()) {
      throw std::runtime_error("Cannot resize beyond capacity. Pre-allocate sufficient capacity.");
    }
    size_ = new_size;
  }
};

} // namespace tnn
