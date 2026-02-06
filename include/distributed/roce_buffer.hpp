/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <infiniband/verbs.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "device/dptr.hpp"
#include "endian.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {

class RoCEBuffer {
private:
  uint8_t *data_;
  size_t size_;
  size_t capacity_;
  ibv_pd *pd_;
  ibv_mr *mr_;
  Endianness endianess_ = get_system_endianness();

  static constexpr size_t MAX_ARRAY_SIZE = static_cast<size_t>(-1) - 8;
  static constexpr size_t DEFAULT_CAPACITY = 10;

  bool in_range(size_t index) const { return index < size_; }

  std::string get_out_of_bound_msg(int index) const {
    return "Array bounds is (0, " + std::to_string(size_) + "), accessed: " + std::to_string(index);
  }

  static size_t huge_capacity(size_t min_capacity) {
    if (min_capacity > MAX_ARRAY_SIZE) {
      throw std::bad_alloc();
    }
    return min_capacity > MAX_ARRAY_SIZE ? static_cast<size_t>(-1) : MAX_ARRAY_SIZE;
  }

  void allocate(size_t new_capacity) {
    constexpr size_t alignment = 4096;  // Page alignment is good for RDMA

    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, new_capacity) != 0) {
      throw std::bad_alloc();
    }
    data_ = static_cast<uint8_t *>(ptr);

    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
    capacity_ = new_capacity;

    if (pd_) {
      mr_ = ibv_reg_mr(pd_, data_, capacity_, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (!mr_) {
        free(data_);
        throw std::runtime_error("Failed to register MR");
      }
    }
  }

  void deallocate() {
    if (mr_) {
      ibv_dereg_mr(mr_);
      mr_ = nullptr;
    }
    if (data_) {
      free(data_);
      data_ = nullptr;
    }
    capacity_ = 0;
  }

public:
  RoCEBuffer(ibv_pd *pd, size_t initial_capacity = 0)
      : data_(nullptr),
        size_(0),
        capacity_(0),
        pd_(pd),
        mr_(nullptr) {
    if (initial_capacity > MAX_ARRAY_SIZE) {
      throw std::invalid_argument("Illegal Capacity: " + std::to_string(initial_capacity));
    }
    if (initial_capacity > 0) {
      allocate(initial_capacity);
    }
  }

  ~RoCEBuffer() { deallocate(); }

  RoCEBuffer(const RoCEBuffer &) = delete;
  RoCEBuffer &operator=(const RoCEBuffer &) = delete;

  RoCEBuffer(RoCEBuffer &&other) noexcept
      : data_(other.data_),
        size_(other.size_),
        capacity_(other.capacity_),
        pd_(other.pd_),
        mr_(other.mr_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.mr_ = nullptr;
  }

  RoCEBuffer &operator=(RoCEBuffer &&other) noexcept {
    if (this != &other) {
      deallocate();
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      pd_ = other.pd_;
      mr_ = other.mr_;

      other.data_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
      other.mr_ = nullptr;
    }
    return *this;
  }

  uint8_t &operator[](size_t index) { return data_[index]; }
  const uint8_t &operator[](size_t index) const { return data_[index]; }

  void set_endianess(Endianness endianess) { endianess_ = endianess; }
  Endianness get_endianess() const { return endianess_; }

  uint8_t *get() { return data_; }
  const uint8_t *get() const { return data_; }

  bool empty() const { return size_ == 0; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  ibv_mr *get_mr() const { return mr_; }
  uint32_t get_lkey() const { return mr_ ? mr_->lkey : 0; }

  void reserve(size_t new_capacity) { ensure_capacity(new_capacity); }
  void clear() { size_ = 0; }

  template <typename T>
  inline void write(size_t &offset, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure_capacity(offset + sizeof(T));
    std::memcpy(data_ + offset, &value, sizeof(T));
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
    std::memcpy(data_ + offset, arr, byte_size);
    if (offset + length * sizeof(T) > size_) {
      size_ = offset + length * sizeof(T);
    }
    offset += length * sizeof(T);
  }

  inline void write(size_t &offset, const dptr &ptr, size_t length) {
    ensure_capacity(offset + length);
    ptr.copy_to_host(data_ + offset, length);
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

  template <typename T>
  inline void read(size_t &offset, T &value) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    if (offset + sizeof(T) > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + sizeof(T)));
    }
    std::memcpy(&value, data_ + offset, sizeof(T));
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
    std::memcpy(arr, data_ + offset, byte_size);
    if (endianess_ != get_system_endianness()) {
      parallel_for<size_t>(0, length, [&](size_t i) { bswap(arr[i]); });
    }
    offset += byte_size;
  }

  inline void read(size_t &offset, dptr &ptr, size_t length) const {
    if (offset + length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + length));
    }
    ptr.copy_from_host(data_ + offset, length);
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
    ensure_capacity(new_size);
    size_ = new_size;
  }

  void fill(uint8_t value) {
    ensure_capacity(size_);
    std::fill(data_, data_ + size_, value);
  }

private:
  void ensure_capacity(size_t min_capacity) {
    if (capacity_ >= min_capacity) {
      return;
    }
    size_t old_capacity = capacity_;
    size_t new_capacity = old_capacity + (old_capacity >> 1);

    if (old_capacity == 0) {
      new_capacity = std::max(DEFAULT_CAPACITY, min_capacity);
    }

    if (new_capacity < min_capacity) {
      new_capacity = min_capacity;
    }

    if (new_capacity > MAX_ARRAY_SIZE) {
      new_capacity = huge_capacity(min_capacity);
    }

    uint8_t *old_data = data_;
    size_t old_size = size_;
    ibv_mr *old_mr = mr_;

    // Allocate new buffer and register it
    allocate(new_capacity);

    if (old_data != nullptr) {
      std::memcpy(data_, old_data, old_size);
      if (old_mr) {
        ibv_dereg_mr(old_mr);
      }
      free(old_data);
    }
    size_ = old_size;
  }
};

}  // namespace tnn
