#pragma once

#include "context.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace tdevice {
// all supported device types
enum class DeviceType { CPU, GPU };

class Device {
public:
  Device(DeviceType type, int id) : type_(type), id_(id) {}
  ~Device() = default;

  const DeviceType &getDeviceType() const { return type_; }

  int getID() const { return id_; }

  std::string getName() const {
    switch (type_) {
    case DeviceType::CPU:
      return "CPU" + std::to_string(id_);
    case DeviceType::GPU:
      return "GPU" + std::to_string(id_);
    default:
      return "UNKNOWN";
    }
  }

  size_t getTotalMemory() const;
  size_t getAvailableMemory() const;

  void *allocateMemory(size_t size) const {}
  void deallocateMemory(void *ptr) const;
  void copyToDevice(void *dest, const void *src, size_t size) const;
  void copyToHost(void *dest, const void *src, size_t size) const;

private:
  DeviceType type_;
  std::vector<Context *> contexts_;
  int id_;
};

} // namespace tdevice