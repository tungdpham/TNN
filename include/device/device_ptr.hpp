#pragma once

#include "device.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tdevice {

template <typename T> class device_ptr {
  static_assert(std::is_trivially_copyable_v<T>, "Type T must be trivially copyable.");

public:
  // Constructors
  explicit device_ptr(T *ptr = nullptr, Device *device = nullptr) : ptr_(ptr), device_(device) {}

  device_ptr(device_ptr &&other) noexcept : ptr_(other.ptr_), device_(other.device_) {
    other.ptr_ = nullptr;
    other.device_ = nullptr;
  }

  device_ptr(const device_ptr &) = delete;

  void reset(T *ptr = nullptr, Device *device = nullptr) {
    if (ptr_) {
      if (device_) {
        device_->deallocateMemory(static_cast<void *>(ptr_));
      } else {
        throw std::runtime_error(
            "Attempting to deallocate device memory without associated device.");
      }
    }
    ptr_ = ptr;
    device_ = device;
  }

  ~device_ptr() { reset(); }

  device_ptr &operator=(device_ptr &&other) noexcept {
    if (this != &other) {
      reset();

      ptr_ = other.ptr_;
      device_ = other.device_;

      other.ptr_ = nullptr;
      other.device_ = nullptr;
    }
    return *this;
  }

  device_ptr &operator=(const device_ptr &) = delete;

  T *release() {
    T *temp = ptr_;
    ptr_ = nullptr;
    device_ = nullptr;
    return temp;
  }

  T *get() const { return ptr_; }
  Device *getDevice() const { return device_; }

  explicit operator bool() const { return ptr_ != nullptr; }

private:
  T *ptr_;
  Device *device_;
};

// template specialization for arrays
template <typename T> class device_ptr<T[]> {
  static_assert(std::is_trivially_copyable_v<T>,
                "Type T must be trivially copyable for array elements.");

public:
  explicit device_ptr(T *ptr = nullptr, Device *device = nullptr, size_t count = 0)
      : ptr_(ptr), device_(device), count_(count) {}

  device_ptr(device_ptr &&other) noexcept
      : ptr_(other.ptr_), device_(other.device_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.device_ = nullptr;
    other.count_ = 0;
  }

  device_ptr(const device_ptr &) = delete;

  void reset(T *ptr = nullptr, Device *device = nullptr, size_t count = 0) {
    if (ptr_) {
      if (device_) {
        device_->deallocateMemory(static_cast<void *>(ptr_));
      } else {
        throw std::runtime_error(
            "Attempting to deallocate device memory without associated device.");
      }
    }
    ptr_ = ptr;
    device_ = device;
    count_ = count; // Set the new count
  }

  ~device_ptr() { reset(); }

  // Operators
  device_ptr &operator=(device_ptr &&other) noexcept {
    if (this != &other) {
      reset();

      ptr_ = other.ptr_;
      device_ = other.device_;
      count_ = other.count_;

      other.ptr_ = nullptr;
      other.device_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  device_ptr &operator=(const device_ptr &) = delete;

  T *release() {
    T *temp = ptr_;
    ptr_ = nullptr;
    device_ = nullptr;
    count_ = 0;
    return temp;
  }

  T *get() const { return ptr_; }
  Device *getDevice() const { return device_; }
  size_t getCount() const { return count_; }

  explicit operator bool() const { return ptr_ != nullptr; }

private:
  T *ptr_;
  Device *device_;
  size_t count_;
};

template <typename T> device_ptr<T> make_ptr(Device *device) {
  static_assert(std::is_trivially_copyable_v<T>, "Type T must be all device-compatible.");

  if (!device) {
    throw std::invalid_argument("Device cannot be null when making pointer");
  }

  T *ptr = static_cast<T *>(device->allocateMemory(sizeof(T)));
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }

  return device_ptr<T>(ptr, device);
}

template <typename T>
typename std::enable_if<std::is_array<T>::value, device_ptr<T>>::type make_array_ptr(Device *device,
                                                                                     size_t count) {
  using ElementT = typename std::remove_extent<T>::type;

  static_assert(std::is_trivially_copyable_v<ElementT>,
                "Array element type must be trivially copyable.");

  if (!device) {
    throw std::invalid_argument("Device cannot be null when making array pointer");
  }
  if (count == 0) {
    return device_ptr<T>(nullptr, nullptr, 0);
  }

  ElementT *ptr = static_cast<ElementT *>(device->allocateMemory(sizeof(ElementT) * count));
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }

  return device_ptr<T>(ptr, device, count);
}

} // namespace tdevice