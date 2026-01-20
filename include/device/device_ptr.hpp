#pragma once

#include "device.hpp"
#include "device_manager.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace tnn {

class device_ptr {
public:
  explicit device_ptr(void *ptr = nullptr, const Device *device = nullptr, size_t count = 0,
                      size_t alignment = 32)
      : ptr_(ptr), device_(device), size_(count), capacity_(count), alignment_(alignment) {}

  device_ptr(std::nullptr_t)
      : ptr_(nullptr), device_(nullptr), size_(0), capacity_(0), alignment_(32) {}

  device_ptr(device_ptr &&other) noexcept
      : ptr_(other.ptr_), device_(other.device_), size_(other.size_), capacity_(other.capacity_),
        alignment_(other.alignment_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.alignment_ = 32;
  }

  device_ptr(const device_ptr &) = delete;

  void reset() noexcept {
    if (ptr_ && device_) {
      device_->deallocateAlignedMemory(static_cast<void *>(ptr_));
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  ~device_ptr() { reset(); }

  // Operators
  device_ptr &operator=(device_ptr &&other) noexcept {
    if (this != &other) {
      reset();

      ptr_ = other.ptr_;
      device_ = other.device_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      alignment_ = other.alignment_;

      other.ptr_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  device_ptr &operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  device_ptr &operator=(const device_ptr &) = delete;

  void *release() {
    void *temp = ptr_;
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    return temp;
  }

  template <typename T> T *get() { return static_cast<T *>(ptr_); }
  template <typename T> const T *get() const { return static_cast<const T *>(ptr_); }

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
      throw std::runtime_error("device_ptr: Bad Alloc");
    }
    if (ptr_) {
      device_->deallocateAlignedMemory(static_cast<void *>(ptr_));
    }
    ptr_ = new_ptr;
    size_ = new_size;
    capacity_ = new_size;
  }

  void ensure(size_t required_count, const Device *device = nullptr) {
    if (device != nullptr && this->device_ != device) {
      reset();
      this->device_ = device;
    }
    if (capacity_ < required_count) {
      resize(required_count);
    }
  }

  explicit operator bool() const { return ptr_ != nullptr; }

private:
  void *ptr_;
  const Device *device_;
  size_t size_;
  size_t capacity_;
  size_t alignment_;
};

template <typename T>
device_ptr make_dptr_t(const Device *device, size_t count, size_t alignment = 64) {
  using ElementT = typename std::remove_extent<T>::type;
  static_assert(std::is_trivially_copyable_v<ElementT>,
                "Array element type must be trivially copyable.");
  if (!device) {
    throw std::invalid_argument("Device cannot be null when making array pointer");
  }
  if (count == 0) {
    return device_ptr(nullptr, device, 0);
  }
  void *ptr = device->allocateAlignedMemory(sizeof(ElementT) * count, alignment);
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }
  return device_ptr(ptr, device, count);
}

inline device_ptr make_dptr(const Device *device, size_t byte_size, size_t alignment = 64) {
  if (!device) {
    throw std::invalid_argument("Device cannot be null when making device pointer");
  }
  if (byte_size == 0)
    return device_ptr(nullptr, device, 0);

  void *ptr = device->allocateAlignedMemory(byte_size, alignment);
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }
  return device_ptr(ptr, device, byte_size);
}

template <typename T> device_ptr to_cpu(const device_ptr &src_ptr) {
  if (!src_ptr.getDevice()) {
    throw std::runtime_error("No associated device to perform to_cpu()");
  }

  if (src_ptr.device_type() == DeviceType::CPU) {
    // Already on CPU, create a copy
    const Device &cpu_device = getCPU();
    auto cpu_ptr = make_dptr_t<T>(&cpu_device, src_ptr.size(), src_ptr.alignment());
    cpu_device.copyToHost(cpu_ptr.template get<T>(), src_ptr.template get<T>(),
                          sizeof(typename std::remove_extent<T>::type) * src_ptr.size());
    return cpu_ptr;
  }

  const Device &cpu_device = getCPU();
  auto cpu_ptr = make_dptr_t<T>(&cpu_device, src_ptr.size(), src_ptr.alignment());
  cpu_device.copyToHost(cpu_ptr.template get<T>(), src_ptr.template get<T>(),
                        sizeof(typename std::remove_extent<T>::type) * src_ptr.size());
  return cpu_ptr;
}

template <typename T> device_ptr to_gpu(const device_ptr &src_ptr, int gpu_id = 0) {
  if (!src_ptr.getDevice()) {
    throw std::runtime_error("No associated device to perform to_gpu()");
  }

  if (src_ptr.device_type() == DeviceType::GPU) {
    // Already on GPU, create a copy
    const Device &gpu_device = getGPU(gpu_id);
    auto gpu_ptr = make_dptr_t<T>(&gpu_device, src_ptr.size(), src_ptr.alignment());
    gpu_device.copyToDevice(gpu_ptr.template get<T>(), src_ptr.template get<T>(),
                            sizeof(typename std::remove_extent<T>::type) * src_ptr.size());
    return gpu_ptr;
  }

  const Device &gpu_device = getGPU(gpu_id);
  auto gpu_ptr = make_dptr_t<T>(&gpu_device, src_ptr.size(), src_ptr.alignment());
  gpu_device.copyToDevice(gpu_ptr.template get<T>(), src_ptr.template get<T>(),
                          sizeof(typename std::remove_extent<T>::type) * src_ptr.size());
  return gpu_ptr;
}

} // namespace tnn