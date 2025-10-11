/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace tpipeline {

class TBuffer {
private:
  uint8_t *data_;
  size_t size_;
  size_t capacity_;

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

public:
  TBuffer() : data_(nullptr), size_(0), capacity_(0) {}

  explicit TBuffer(size_t initial_capacity) : data_(nullptr), size_(0), capacity_(0) {
    if (initial_capacity > MAX_ARRAY_SIZE) {
      throw std::invalid_argument("Illegal Capacity: " + std::to_string(initial_capacity));
    }
    if (initial_capacity > 0) {
      data_ = static_cast<uint8_t *>(malloc(initial_capacity));
      if (data_ == nullptr) {
        throw std::bad_alloc();
      }
      capacity_ = initial_capacity;
    }
  }

  TBuffer(std::initializer_list<uint8_t> init) : data_(nullptr), size_(0), capacity_(0) {
    if (init.size() > 0) {
      data_ = static_cast<uint8_t *>(malloc(init.size()));
      if (data_ == nullptr) {
        throw std::bad_alloc();
      }
      capacity_ = init.size();
      size_ = init.size();
      size_t i = 0;
      for (const auto &val : init) {
        data_[i++] = val;
      }
    }
  }

  TBuffer(const TBuffer &other) : data_(nullptr), size_(0), capacity_(0) {
    if (other.size_ > 0) {
      data_ = static_cast<uint8_t *>(malloc(other.capacity_));
      if (data_ == nullptr) {
        throw std::bad_alloc();
      }
      capacity_ = other.capacity_;
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

  static TBuffer from_existing_pointer(uint8_t *data, size_t size) {
    if (data == nullptr && size > 0) {
      throw std::invalid_argument("Data pointer is null but size is greater than zero");
    }
    TBuffer buffer;
    buffer.data_ = data;
    buffer.size_ = size;
    buffer.capacity_ = size;
    return buffer;
  }

  ~TBuffer() {
    if (data_ != nullptr) {
      free(data_);
      data_ = nullptr;
    }
  }

  TBuffer &operator=(const TBuffer &other) {
    if (this != &other) {
      free(data_);
      data_ = nullptr;
      size_ = 0;
      capacity_ = 0;

      if (other.size_ > 0) {
        data_ = static_cast<uint8_t *>(malloc(other.capacity_));
        if (data_ == nullptr) {
          throw std::bad_alloc();
        }
        capacity_ = other.capacity_;
        size_ = other.size_;
        std::memcpy(data_, other.data_, size_);
      }
    }
    return *this;
  }

  TBuffer &operator=(TBuffer &&other) noexcept {
    if (this != &other) {
      free(data_);
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

  uint8_t *get() { return data_; }
  const uint8_t *get() const { return data_; }

  bool empty() const { return size_ == 0; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  void reserve(size_t new_capacity) { ensure_capacity(new_capacity); }

  void shrink_to_fit() {
    if (size_ < capacity_) {
      if (size_ == 0) {
        free(data_);
        data_ = nullptr;
        capacity_ = 0;
      } else {
        uint8_t *new_data = static_cast<uint8_t *>(malloc(size_));
        if (new_data == nullptr) {
          throw std::bad_alloc();
        }
        std::memcpy(new_data, data_, size_);
        free(data_);
        data_ = new_data;
        capacity_ = size_;
      }
    }
  }

  void clear() { size_ = 0; }

  template <typename T> void write_value(const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    ensure_capacity(size_ + sizeof(T));
    std::memcpy(data_ + size_, &value, sizeof(T));
    size_ += sizeof(T);
  }

  void write_value(const std::string &str) {
    uint64_t str_length = static_cast<uint64_t>(str.size());
    write_value(str_length);
    if (str_length > 0) {
      const char *chars = str.data();
      write_array(reinterpret_cast<const uint8_t *>(chars), str_length);
    }
  }

  template <typename T> void write_array(const T *arr, size_t length) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    ensure_capacity(size_ + byte_size);
    std::memcpy(data_ + size_, arr, byte_size);
    size_ += byte_size;
  }

  template <typename T> T read_value(size_t &offset) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    if (offset + sizeof(T) > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + sizeof(T)));
    }
    T value;
    std::memcpy(&value, data_ + offset, sizeof(T));
    offset += sizeof(T);
    return value;
  }

  std::string read_string(size_t &offset) const {
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

  template <typename T> void read_array(size_t &offset, T *arr, size_t length) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable (primitive or POD type)");
    size_t byte_size = sizeof(T) * length;
    if (offset + byte_size > size_) {
      throw std::out_of_range(get_out_of_bound_msg(offset + byte_size));
    }
    std::memcpy(arr, data_ + offset, byte_size);
    offset += byte_size;
  }

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

    uint8_t *new_data = static_cast<uint8_t *>(malloc(new_capacity));
    if (new_data == nullptr) {
      throw std::bad_alloc();
    }
    if (data_ != nullptr) {
      std::memcpy(new_data, data_, size_);
      free(data_);
    }
    data_ = new_data;
    capacity_ = new_capacity;
  }
};
} // namespace tpipeline
