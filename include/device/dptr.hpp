#pragma once

#include "device.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace tnn {

constexpr size_t DEFAULT_ALIGNMENT = 64;

// View class for device pointer, does not own the memory
class dptr_view {
protected:
  const Device *device_;
  void *ptr_;
  size_t size_;
  size_t capacity_;
  size_t alignment_;

public:
  dptr_view(const Device *device = nullptr, void *ptr = nullptr, size_t size = 0,
            size_t alignment = DEFAULT_ALIGNMENT)
      : device_(device), ptr_(ptr), size_(size), capacity_(size), alignment_(alignment) {}

  dptr_view(std::nullptr_t)
      : device_(nullptr), ptr_(nullptr), size_(0), capacity_(0), alignment_(DEFAULT_ALIGNMENT) {}

  // enable copy for view
  dptr_view(const dptr_view &other)
      : device_(other.device_), ptr_(other.ptr_), size_(other.size_), capacity_(other.capacity_),
        alignment_(other.alignment_) {}

  dptr_view &operator=(const dptr_view &other) {
    if (this != &other) {
      device_ = other.device_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      alignment_ = other.alignment_;
    }
    return *this;
  }

  dptr_view(dptr_view &&other) noexcept
      : device_(other.device_), ptr_(other.ptr_), size_(other.size_), capacity_(other.capacity_),
        alignment_(other.alignment_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  dptr_view &operator=(dptr_view &&other) noexcept {
    if (this != &other) {
      device_ = other.device_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      alignment_ = other.alignment_;

      other.ptr_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  dptr_view &operator=(std::nullptr_t) {
    device_ = nullptr;
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    alignment_ = DEFAULT_ALIGNMENT;
    return *this;
  }

  // ! DO NOT USE IN PLACE OPERATOR (+=, -=), THEY WILL MODIFY THE ORIGINAL dptr_view

  dptr_view operator+(size_t offset) const {
    if (offset > size_) {
      throw std::out_of_range("dptr_view::operator+: offset out of range");
    }
    return dptr_view(device_, static_cast<void *>(static_cast<char *>(ptr_) + offset),
                     size_ - offset, alignment_);
  }

  dptr_view operator-(size_t offset) const {
    if (offset > size_) {
      throw std::out_of_range("dptr_view::operator-: offset out of range");
    }
    return dptr_view(device_, static_cast<void *>(static_cast<char *>(ptr_) - offset),
                     size_ - offset, alignment_);
  }

  template <typename T> T *get() { return static_cast<T *>(ptr_); }

  template <typename T> const T *get() const { return static_cast<const T *>(ptr_); }

  const Device *getDevice() const { return device_; }
};

class dptr : public dptr_view {
public:
  using dptr_view::dptr_view;

  explicit dptr(const Device *device = nullptr, void *ptr = nullptr, size_t size = 0,
                size_t alignment = DEFAULT_ALIGNMENT)
      : dptr_view(device, ptr, size, alignment) {}

  dptr(std::nullptr_t) : dptr_view(nullptr) {}

  void reset() noexcept {
    if (ptr_ && device_) {
      device_->deallocateAlignedMemory(static_cast<void *>(ptr_));
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  ~dptr() { reset(); }

  dptr(const dptr &) = delete;

  dptr &operator=(const dptr &) = delete;

  dptr(dptr &&other) noexcept
      : dptr_view(other.device_, other.ptr_, other.size_, other.alignment_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  dptr &operator=(dptr &&other) noexcept {
    if (this != &other) {
      reset();
      device_ = other.device_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      alignment_ = other.alignment_;

      other.ptr_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  const Device *getDevice() const { return device_; }

  DeviceType device_type() const {
    if (!device_) {
      throw std::runtime_error("No associated device to get device type from.");
    }
    return device_->device_type();
  }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  size_t alignment() const { return alignment_; }

  void resize(size_t new_size) {
    void *new_ptr = device_->allocateAlignedMemory(new_size, alignment_);
    if (!new_ptr) {
      throw std::runtime_error("dptr: Bad Alloc");
    }
    if (ptr_) {
      device_->deallocateAlignedMemory(static_cast<void *>(ptr_));
    }
    ptr_ = new_ptr;
    size_ = new_size;
    capacity_ = new_size;
  }

  void ensure(size_t required_count) {
    if (capacity_ < required_count) {
      resize(required_count);
    }
  }

  void copy_to_host(void *host_ptr, size_t byte_size) const {
    if (!device_) {
      throw std::runtime_error("No associated device to perform copy_to_host()");
    }
    device_->copyToHost(host_ptr, ptr_, byte_size);
  }

  void copy_from_host(const void *host_ptr, size_t byte_size) {
    if (!device_) {
      throw std::runtime_error("No associated device to perform copy_from_host()");
    }
    device_->copyToDevice(ptr_, host_ptr, byte_size);
  }

  explicit operator bool() const { return ptr_ != nullptr; }
};

template <typename T> dptr make_dptr_t(const Device *device, size_t count, size_t alignment = 64) {
  using ElementT = typename std::remove_extent<T>::type;
  static_assert(std::is_trivially_copyable_v<ElementT>,
                "Array element type must be trivially copyable.");
  if (!device) {
    throw std::invalid_argument("Device cannot be null when making array pointer");
  }
  if (count == 0) {
    return dptr(device, nullptr, 0);
  }
  void *ptr = device->allocateAlignedMemory(sizeof(ElementT) * count, alignment);
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }
  return dptr(device, ptr, sizeof(ElementT) * count, alignment);
}

inline dptr make_dptr(const Device *device, size_t byte_size, size_t alignment = 64) {
  if (!device) {
    throw std::invalid_argument("Device cannot be null when making device pointer");
  }
  if (byte_size == 0)
    return dptr(device, nullptr, 0);

  void *ptr = device->allocateAlignedMemory(byte_size, alignment);
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }
  return dptr(device, ptr, byte_size, alignment);
}

} // namespace tnn