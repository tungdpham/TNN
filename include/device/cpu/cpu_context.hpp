#pragma once

#include "device/context.hpp"

#include <cstddef>

namespace tdevice {
class CPUContext : public Context {
public:
  explicit CPUContext();

  size_t getTotalMemory() const override;
  size_t getAvailableMemory() const override;
  void *allocateMemory(size_t size) override;
  void deallocateMemory(void *ptr) override;
  void copyToDevice(void *dest, const void *src, size_t size) override;
  void copyToHost(void *dest, const void *src, size_t size) override;
};
} // namespace tdevice