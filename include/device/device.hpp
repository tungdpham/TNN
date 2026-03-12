/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstring>
#include <memory>
#include <string>

#include "context.hpp"
#include "device/flow.hpp"
#include "device_type.hpp"

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

  bool operator==(const Device &other) const;

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
  void createFlow(flowHandle_t handle) const;
  Flow *getFlow(flowHandle_t handle) const;
  Context *context() const { return context_.get(); }

private:
  DeviceType type_;
  int id_;
  std::unique_ptr<Context> context_;
};

}  // namespace tnn