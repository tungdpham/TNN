#include "device/device.hpp"
#include "device/context.hpp"

namespace tnn {
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

const DeviceType &Device::device_type() const { return type_; }

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

void *Device::allocateAlignedMemory(size_t size, size_t alignment) const {
  return context_->allocateAlignedMemory(size, alignment);
}

void Device::deallocateAlignedMemory(void *ptr) const { context_->deallocateAlignedMemory(ptr); }

void Device::copyToDevice(void *dest, const void *src, size_t size) const {
  context_->copyToDevice(dest, src, size);
}

void Device::copyToHost(void *dest, const void *src, size_t size) const {
  context_->copyToHost(dest, src, size);
}

void Device::createFlow(const std::string &flow_id) const { context_->createFlow(flow_id); }

Flow *Device::getFlow(const std::string &flow_id) const { return context_->getFlow(flow_id); }

} // namespace tnn
