#pragma once

#include "device/allocator.hpp"
#include "device/dptr.hpp"

namespace tnn {

// default allocator
class DeviceAllocator : public IAllocator {
public:
  DeviceAllocator(const Device& device) : device_(device) {}

  dptr allocate(size_t size) override {
    void* ptr = device_.allocateAlignedMemory(size, DEFAULT_ALIGNMENT);
    auto device_storage = std::make_shared<device_storage>(&device_, ptr, size, DEFAULT_ALIGNMENT);
    return dptr(device_storage, 0, size);
  }

private:
  const Device& device_;
};

}  // namespace tnn