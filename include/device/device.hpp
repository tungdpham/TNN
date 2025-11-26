#pragma once

#include "context.hpp"
#include "device_type.hpp"

#include <cstring>
#include <memory>
#include <string>

namespace tnn {

class Device {
public:
  Device(DeviceType type, int id, std::unique_ptr<Context> context);
  ~Device();

  // Move constructor and assignment operator
  Device(Device &&other) noexcept;
  Device &operator=(Device &&other) noexcept;

  // Explicitly delete copy constructor and copy assignment operator
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  const DeviceType &device_type() const;
  int getID() const;
  std::string getName() const;
  size_t getTotalMemory() const;
  size_t getAvailableMemory() const;
  void *allocateMemory(size_t size) const;
  void deallocateMemory(void *ptr) const;
  void *allocateAlignedMemory(size_t size, size_t alignment) const;
  void deallocateAlignedMemory(void *ptr) const;
  void copyToDevice(void *dest, const void *src, size_t size) const;
  void copyToHost(void *dest, const void *src, size_t size) const;
  void createFlow(const std::string &flow_id) const;
  Flow *getFlow(const std::string &flow_id) const;

private:
  DeviceType type_;
  int id_;
  std::unique_ptr<Context> context_;
};

} // namespace tnn