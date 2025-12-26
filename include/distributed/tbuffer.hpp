/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "endian.hpp"
#include "threading/thread_handler.hpp"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "ops/cpu/kernels.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <type_traits>
namespace tnn {

class TBuffer {
private:
  uint8_t *data_;
  size_t size_;
  size_t capacity_;
  // Endianess of the data in the buffer, does not apply to write. Read functions will swap bytes if
  // needed.
  Endianness endianess_ = get_system_endianness(); // 0 for little-endian and 1 for big-endian

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
    constexpr size_t alignment = 64; // 64-byte alignment for AVX2
#ifdef _WIN32
    data_ = static_cast<uint8_t *>(_aligned_malloc(new_capacity, alignment));
#else
    // POSIX aligned_alloc requires size to be a multiple of alignment
    size_t adjusted_size = ((new_capacity + alignment - 1) / alignment) * alignment;
    data_ = static_cast<uint8_t *>(std::aligned_alloc(alignment, adjusted_size));
#endif
    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
    capacity_ = new_capacity;
  }

  void deallocate() {
#ifdef _WIN32
    _aligned_free(data_);
#else
    free(data_);
#endif
    data_ = nullptr;
    capacity_ = 0;
  }

public:
  TBuffer() : data_(nullptr), size_(0), capacity_(0) {}

  explicit TBuffer(size_t initial_capacity) : data_(nullptr), size_(0), capacity_(0) {
    if (initial_capacity > MAX_ARRAY_SIZE) {
      throw std::invalid_argument("Illegal Capacity: " + std::to_string(initial_capacity));
    }
    if (initial_capacity > 0) {
      allocate(initial_capacity);
    }
  }

  TBuffer(std::initializer_list<uint8_t> init) : data_(nullptr), size_(0), capacity_(0) {
    if (init.size() > 0) {
      allocate(init.size());
      std::memcpy(data_, init.begin(), init.size());
      size_ = init.size();
    }
  }

  TBuffer(const TBuffer &other) : data_(nullptr), size_(0), capacity_(0) {
    if (other.size_ > 0) {
      allocate(other.size_);
      size_ = other.size_;
      std::memcpy(data_, other.data_, size_);
    }
  }

  TBuffer(TBuffer &&other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  ~TBuffer() { deallocate(); }

  TBuffer &operator=(const TBuffer &other) {
    if (this != &other) {
      deallocate();

      if (other.size_ > 0) {
        allocate(other.size_);
        size_ = other.size_;
        std::memcpy(data_, other.data_, size_);
      }
    }
    return *this;
  }

  TBuffer &operator=(TBuffer &&other) noexcept {
    if (this != &other) {
      deallocate();
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
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

  void reserve(size_t new_capacity) { ensure_capacity(new_capacity); }

  void clear() { size_ = 0; }

#if defined(ARCH_64)
  template <typename T> inline void write_value(const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure_capacity(size_ + sizeof(T));
    std::memcpy(data_ + size_, &value, sizeof(T));
    size_ += sizeof(T);
  }

  template <typename T>
  inline void write_array(const T *arr, size_t length, bool parallel = false) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure_capacity(size_ + length * sizeof(T));
    if (!parallel) {
      std::copy(arr, arr + length, reinterpret_cast<T *>(data_ + size_));
    } else {
      size_t num_blocks = get_num_threads();
      size_t block_size = length / num_blocks;
      parallel_for<size_t>(0, num_blocks, [&](size_t block_idx) {
        size_t start = block_idx * block_size;
        size_t end = std::min(length, (block_idx + 1) * block_size);
        std::copy(arr + start, arr + end, reinterpret_cast<T *>(data_ + size_) + start);
      });
    }
    size_ += length * sizeof(T);
  }

  inline void write_string(const std::string &str) {
    uint64_t str_length = static_cast<uint64_t>(str.size());
    write_value<uint64_t>(str_length);
    if (str_length > 0) {
      write_array(reinterpret_cast<const uint8_t *>(str.data()), str_length);
    }
  }

  template <typename T> inline T read_value(size_t &offset) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    if (offset + sizeof(T) > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + sizeof(T)));
    }
    T value;
    std::memcpy(&value, data_ + offset, sizeof(T));
    offset += sizeof(T);
    if (endianess_ != get_system_endianness()) {
      bswap(value);
    }
    return value;
  }

  template <typename T>
  inline void read_array(size_t &offset, T *arr, size_t length, bool parallel = false) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    if (offset + byte_size > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + byte_size));
    }
    if (!parallel) {
      std::copy(data_ + offset, data_ + offset + byte_size, reinterpret_cast<uint8_t *>(arr));
    } else {
      size_t num_blocks = get_num_threads();
      size_t block_size = byte_size / num_blocks;
      parallel_for<size_t>(0, num_blocks, [&](size_t block_idx) {
        size_t start = block_idx * block_size;
        size_t end = std::min(byte_size, (block_idx + 1) * block_size);
        std::copy(data_ + offset + start, data_ + offset + end,
                  reinterpret_cast<uint8_t *>(arr) + start);
      });
    }
    // should not happen often
    if (endianess_ != get_system_endianness()) {
      parallel_for<size_t>(0, length, [&](size_t i) { bswap(arr[i]); });
    }
    offset += byte_size;
  }

  inline std::string read_string(size_t &offset) const {
    uint64_t str_length = read_value<uint64_t>(offset);
    if (offset + str_length > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + str_length));
    }
    std::string str;
    if (str_length > 0) {
      str.resize(str_length);
      std::memcpy(&str[0], data_ + offset, str_length);
      offset += str_length;
    }
    return str;
  }
#elif defined(ARCH_32)
  // Implementations for 32-bit architectures can go here

#else
#error "Unknown architecture. Codebase supports only 32-bit and 64-bit architectures."
#endif

  void resize(size_t new_size) {
    ensure_capacity(new_size);
    size_ = new_size;
  }

  void fill(uint8_t value) {
    ensure_capacity(size_);
    std::fill(data_, data_ + size_, value); // Fill only up to size, not capacity
  }

private:
  void ensure_capacity(size_t min_capacity) {
    if (capacity_ >= min_capacity) {
      return;
    }
    // grow by 1.5x if possible
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

    allocate(new_capacity);

    if (old_data != nullptr) {
      std::memcpy(data_, old_data, old_size);
#ifdef _WIN32
      _aligned_free(old_data);
#else
      free(old_data);
#endif
    }
    size_ = old_size;
  }
};

} // namespace tnn
