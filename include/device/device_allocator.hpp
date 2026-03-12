/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <unordered_map>

#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"

namespace tnn {

// default allocator
class DeviceAllocator : public IAllocator {
public:
  DeviceAllocator(const Device& device)
      : device_(device) {}

  static DeviceAllocator& instance(const Device& device) {
    static std::mutex registry_mutex;
    static std::unordered_map<const Device*, std::unique_ptr<DeviceAllocator>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (registry.find(&device) == registry.end()) {
      registry[&device] = std::make_unique<DeviceAllocator>(device);
    }
    return *registry[&device];
  }

  dptr allocate(size_t size) override {
    void* ptr = device_->allocateAlignedMemory(size, DEFAULT_ALIGNMENT);
    device_storage* storage_ptr = new device_storage(*device_, ptr, size, DEFAULT_ALIGNMENT);
    auto storage = std::shared_ptr<device_storage>(storage_ptr, [this](device_storage* storage) {
      if (storage) {
        device_->deallocateAlignedMemory(storage->data());
        delete storage;
      }
    });
    return dptr(storage, 0, size);
  }

  void clear() override {
    // no-op since we don't cache any memory blocks
  }

  void reserve(size_t size) override {
    // no-op since we don't cache any memory blocks
  }

  const Device& device() const override { return *device_; }

private:
  csref<Device> device_;
};

inline DeviceAllocator& HostAllocator() { return DeviceAllocator::instance(getHost()); }

inline DeviceAllocator& GPUAllocator(int device_id = 0) {
  return DeviceAllocator::instance(getGPU(device_id));
}

}  // namespace tnn