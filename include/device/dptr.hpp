/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "device/device.hpp"
#include "device/device_type.hpp"
#include "device/sref.hpp"

namespace tnn {

constexpr size_t DEFAULT_ALIGNMENT = 256;

struct device_storage {
private:
  csref<Device> device_;
  void *ptr_;
  size_t capacity_;
  size_t alignment_;

public:
  device_storage(csref<Device> device, void *ptr, size_t capacity, size_t alignment)
      : device_(device),
        ptr_(ptr),
        capacity_(capacity),
        alignment_(alignment) {}

  csref<Device> device() const { return device_; }
  void *data() const { return ptr_; }
  size_t capacity() const { return capacity_; }
  size_t alignment() const { return alignment_; }
};

// device pointer. Shares ownership of device storage. True owners have offset 0.
class dptr {
protected:
  std::shared_ptr<device_storage> storage_;
  size_t offset_;
  size_t capacity_;

public:
  dptr(std::shared_ptr<device_storage> storage = nullptr, size_t offset = 0, size_t capacity = 0)
      : storage_(storage),
        offset_(offset),
        capacity_(capacity) {
    if (storage_ && (offset_ + capacity_ > storage_->capacity())) {
      throw std::out_of_range("Bro what? dptr offset out of range in dptr constructor");
    }
  }

  dptr(std::nullptr_t)
      : storage_(nullptr),
        offset_(0),
        capacity_(0) {}

  operator bool() const { return storage_ != nullptr && storage_->data() != nullptr; }

  const Device &getDevice() const { return storage_->device(); }

  DeviceType device_type() const {
    if (!storage_) {
      return DeviceType::NULL_DEVICE;
    }
    return storage_->device()->device_type();
  }

  size_t capacity() const { return capacity_; }

  size_t alignment() const {
    if (!storage_) {
      return 0;
    }
    return storage_->alignment();
  }

  template <typename T = void>
  T *get() {
    if (!storage_) {
      return nullptr;
    }
    return static_cast<T *>(
        static_cast<void *>(static_cast<uint8_t *>(storage_->data()) + offset_));
  }

  template <typename T = void>
  const T *get() const {
    if (!storage_) {
      return nullptr;
    }
    return static_cast<const T *>(
        static_cast<const void *>(static_cast<const uint8_t *>(storage_->data()) + offset_));
  }

  dptr span(size_t offset, size_t span_size) {
    if (offset + span_size > capacity_) {
      throw std::out_of_range("dptr span size out of range");
    }
    return dptr(storage_, offset_ + offset, span_size);
  }

  const dptr span(size_t offset, size_t span_size) const {
    if (offset + span_size > capacity_) {
      throw std::out_of_range("dptr span size out of range");
    }
    return dptr(storage_, offset_ + offset, span_size);
  }

  dptr operator+(size_t offset) const {
    if (offset > capacity_) {
      throw std::out_of_range("dptr offset out of range");
    }
    return dptr(storage_, offset_ + offset, capacity_ - offset);
  }

  void copy_to_host(void *host_ptr, size_t byte_size) const {
    if (!storage_) {
      throw std::runtime_error("Invalid device storage or device in dptr::copy_to_host");
    }
    if (byte_size > capacity_) {
      throw std::out_of_range("dptr copy_to_host out of range");
    }
    storage_->device()->copyToHost(host_ptr, static_cast<uint8_t *>(storage_->data()) + offset_,
                                   byte_size);
  }

  void copy_from_host(const void *host_ptr, size_t byte_size) {
    if (!storage_) {
      throw std::runtime_error("Invalid device storage or device in dptr::copy_from_host");
    }
    if (byte_size > capacity_) {
      throw std::out_of_range("dptr copy_from_host out of range");
    }
    storage_->device()->copyToDevice(static_cast<uint8_t *>(storage_->data()) + offset_, host_ptr,
                                     byte_size);
  }
};

inline dptr make_dptr(csref<Device> device, size_t byte_size,
                      size_t alignment = DEFAULT_ALIGNMENT) {
  if (byte_size == 0) {
    return dptr(nullptr);
  }
  void *ptr = device->allocateAlignedMemory(byte_size, alignment);
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }
  auto storage = std::shared_ptr<device_storage>(
      new device_storage(device, ptr, byte_size, alignment), [device](device_storage *s) {
        device->deallocateAlignedMemory(s->data());
        delete s;
      });
  return dptr(storage, 0, byte_size);
}

template <typename T>
inline dptr make_dptr_t(csref<Device> device, size_t count, size_t alignment = DEFAULT_ALIGNMENT) {
  return make_dptr(device, count * sizeof(T), alignment);
}

}  // namespace tnn