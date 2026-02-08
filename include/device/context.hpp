#pragma once

#include <cstddef>
#include <cstdlib>

#include "flow.hpp"

namespace tnn {
class Context {
public:
  Context() = default;
  virtual ~Context() = default;

  virtual size_t getTotalMemory() const = 0;
  virtual size_t getAvailableMemory() const = 0;
  virtual void *allocateMemory(size_t size) = 0;
  virtual void deallocateMemory(void *ptr) = 0;
  virtual void *allocateAlignedMemory(size_t size, size_t alignment) = 0;
  virtual void deallocateAlignedMemory(void *ptr) = 0;
  virtual void copyToDevice(void *dest, const void *src, size_t size) = 0;
  virtual void copyToHost(void *dest, const void *src, size_t size) = 0;
  virtual void createFlow(flowHandle_t flow_id) = 0;
  virtual Flow *getFlow(flowHandle_t flow_id) = 0;
};
}  // namespace tnn
