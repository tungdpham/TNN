#include "device/device.hpp"
#include "device/context.hpp"

#include "device/contexts/cpu_context.hpp"
#ifdef USE_CUDA
#include "device/contexts/cuda_context.hpp"
#endif

#include <stdexcept>

namespace tdevice {
Device::Device(DeviceType type, int id, std::unique_ptr<Context> context)
    : type_(type), id_(id), context_(std::move(context)) {}

Device::~Device() = default;

Device::Device(Device &&other) noexcept
    : type_(other.type_), id_(other.id_), context_(std::move(other.context_)) {}

Device &Device::operator=(Device &&other) noexcept {
  if (this != &other) {
    type_ = other.type_;
    id_ = other.id_;
    context_ = std::move(other.context_);
  }
  return *this;
}

const DeviceType &Device::getDeviceType() const { return type_; }

int Device::getID() const { return id_; }

std::string Device::getName() const {
  switch (type_) {
  case DeviceType::CPU:
    return "CPU Device " + std::to_string(id_);
  case DeviceType::GPU:
    return "GPU Device " + std::to_string(id_);
  default:
    return "Unknown Device";
  }
}

size_t Device::getTotalMemory() const { return context_->getTotalMemory(); }

size_t Device::getAvailableMemory() const { return context_->getAvailableMemory(); }

void *Device::allocateMemory(size_t size) const { return context_->allocateMemory(size); }

void Device::deallocateMemory(void *ptr) const { context_->deallocateMemory(ptr); }

void Device::copyToDevice(void *dest, const void *src, size_t size) const {
  context_->copyToDevice(dest, src, size);
}

void Device::copyToHost(void *dest, const void *src, size_t size) const {
  context_->copyToHost(dest, src, size);
}
} // namespace tdevice
