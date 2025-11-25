#pragma once

#include "device.hpp"
#include "device_manager.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace tnn {

template <typename T> class device_ptr {
  static_assert(std::is_trivially_copyable_v<T>, "Type T must be trivially copyable.");

public:
  // Constructors
  explicit device_ptr(T *ptr = nullptr, const Device *device = nullptr)
      : ptr_(ptr), device_(device) {}

  device_ptr(device_ptr &&other) noexcept : ptr_(other.ptr_), device_(other.device_) {
    other.ptr_ = nullptr;
  }

  device_ptr(const device_ptr &) = delete;

  void reset(T *ptr = nullptr, const Device *device = nullptr) {
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
  const Device *getDevice() const { return device_; }

  DeviceType getDeviceType() const {
    if (!device_) {
      throw std::runtime_error("No associated device to get device type from.");
    }
    return device_->getDeviceType();
  }

  explicit operator bool() const { return ptr_ != nullptr; }

private:
  T *ptr_;
  const Device *device_;
};

// template specialization for arrays
template <typename T> class device_ptr<T[]> {
  static_assert(std::is_trivially_copyable_v<T>,
                "Type T must be trivially copyable for array elements.");

public:
  explicit device_ptr(T *ptr = nullptr, const Device *device = nullptr, size_t count = 0,
                      size_t alignment = 32)
      : ptr_(ptr), device_(device), count_(count), alignment_(alignment) {}

  device_ptr(std::nullptr_t) : ptr_(nullptr), device_(nullptr), count_(0), alignment_(32) {}

  device_ptr(device_ptr &&other) noexcept
      : ptr_(other.ptr_), device_(other.device_), count_(other.count_),
        alignment_(other.alignment_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  device_ptr(const device_ptr &) = delete;

  void reset(T *ptr = nullptr, const Device *device = nullptr, size_t count = 0,
             size_t alignment = 32) {
    if (ptr_) {
      if (device_) {
        device_->deallocateAlignedMemory(static_cast<void *>(ptr_));
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
      alignment_ = other.alignment_;

      other.ptr_ = nullptr;
      other.count_ = 0;
      other.alignment_ = 0;
    }
    return *this;
  }

  device_ptr &operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  device_ptr &operator=(const device_ptr &) = delete;

  T *release() {
    T *temp = ptr_;
    ptr_ = nullptr;
    count_ = 0;
    alignment_ = 0;
    return temp;
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

  const Device *getDevice() const { return device_; }

  DeviceType getDeviceType() const {
    if (!device_) {
      throw std::runtime_error("No associated device to get device type from.");
    }
    return device_->getDeviceType();
  }

  size_t size() const { return count_; }

  size_t getAlignment() const { return alignment_; }

  void resize(size_t new_count) {
    T *new_ptr =
        static_cast<T *>(device_->allocateAlignedMemory(sizeof(T) * new_count, alignment_));
    if (!new_ptr) {
      throw std::runtime_error("Bad Alloc");
    }
    if (ptr_) {
      device_->deallocateAlignedMemory(static_cast<void *>(ptr_));
    }
    ptr_ = new_ptr;
    count_ = new_count;
  }

  void ensure(size_t required_count) {
    if (count_ < required_count) {
      resize(required_count);
    }
  }

  explicit operator bool() const { return ptr_ != nullptr; }

private:
  T *ptr_;
  const Device *device_;
  size_t count_;
  size_t alignment_;
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
typename std::enable_if<std::is_array<T>::value, device_ptr<T>>::type
make_array_ptr(const Device *device, size_t count, size_t alignment = 32) {
  using ElementT = typename std::remove_extent<T>::type;

  static_assert(std::is_trivially_copyable_v<ElementT>,
                "Array element type must be trivially copyable.");

  if (!device) {
    throw std::invalid_argument("Device cannot be null when making array pointer");
  }

  if (count == 0) {
    return device_ptr<T>(nullptr, device, 0);
  }

  ElementT *ptr =
      static_cast<ElementT *>(device->allocateAlignedMemory(sizeof(ElementT) * count, alignment));
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }

  return device_ptr<T>(ptr, device, count);
}

template <typename T>
typename std::enable_if<std::is_array<T>::value, device_ptr<T>>::type
to_cpu(const device_ptr<T> &src_ptr) {
  if (!src_ptr.getDevice()) {
    throw std::runtime_error("No associated device to perform to_cpu()");
  }

  if (src_ptr.getDeviceType() == DeviceType::CPU) {
    // Already on CPU, create a copy
    const Device &cpu_device = getCPU();
    auto cpu_ptr = make_array_ptr<T>(&cpu_device, src_ptr.getCount(), src_ptr.getAlignment());
    cpu_device.copyToHost(cpu_ptr.get(), src_ptr.get(),
                          sizeof(typename std::remove_extent<T>::type) * src_ptr.getCount());
    return cpu_ptr;
  }

  const Device &cpu_device = getCPU();
  auto cpu_ptr = make_array_ptr<T>(&cpu_device, src_ptr.getCount(), src_ptr.getAlignment());
  cpu_device.copyToHost(cpu_ptr.get(), src_ptr.get(),
                        sizeof(typename std::remove_extent<T>::type) * src_ptr.getCount());
  return cpu_ptr;
}

template <typename T>
typename std::enable_if<std::is_array<T>::value, device_ptr<T>>::type
to_gpu(const device_ptr<T> &src_ptr, int gpu_id = 0) {
  if (!src_ptr.getDevice()) {
    throw std::runtime_error("No associated device to perform to_gpu()");
  }

  if (src_ptr.getDeviceType() == DeviceType::GPU) {
    // Already on GPU, create a copy
    const Device &gpu_device = getGPU(gpu_id);
    auto gpu_ptr = make_array_ptr<T>(&gpu_device, src_ptr.getCount(), src_ptr.getAlignment());
    gpu_device.copyToDevice(gpu_ptr.get(), src_ptr.get(),
                            sizeof(typename std::remove_extent<T>::type) * src_ptr.getCount());
    return gpu_ptr;
  }

  const Device &gpu_device = getGPU(gpu_id);
  auto gpu_ptr = make_array_ptr<T>(&gpu_device, src_ptr.getCount(), src_ptr.getAlignment());
  gpu_device.copyToDevice(gpu_ptr.get(), src_ptr.get(),
                          sizeof(typename std::remove_extent<T>::type) * src_ptr.getCount());
  return gpu_ptr;
}

} // namespace tnn