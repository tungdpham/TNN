#pragma once

#include <cstddef>
#include <cstdlib>

namespace tnn {
class Context {
public:
  Context() = default;
  ~Context() = default;

  virtual size_t getTotalMemory() const = 0;
  virtual size_t getAvailableMemory() const = 0;
  virtual void *allocateMemory(size_t size) = 0;
  virtual void deallocateMemory(void *ptr) = 0;
  virtual void *allocateAlignedMemory(size_t size, size_t alignment) = 0;
  virtual void deallocateAlignedMemory(void *ptr) = 0;
  virtual void copyToDevice(void *dest, const void *src, size_t size) = 0;
  virtual void copyToHost(void *dest, const void *src, size_t size) = 0;
};
} // namespace tnn
