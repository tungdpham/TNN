#pragma once

#include <unordered_map>

#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"

namespace tnn {

// default allocator
class DeviceAllocator : public IAllocator {
public:
  DeviceAllocator(const Device& device) : device_(device) {}

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
    auto storage = std::make_shared<device_storage>(device_, ptr, size, DEFAULT_ALIGNMENT);
    return dptr(storage, 0, size);
  }

private:
  csref<Device> device_;
};

inline DeviceAllocator& HostAllocator() { return DeviceAllocator::instance(getCPU()); }

inline DeviceAllocator& GPUAllocator() { return DeviceAllocator::instance(getGPU()); }

}  // namespace tnn