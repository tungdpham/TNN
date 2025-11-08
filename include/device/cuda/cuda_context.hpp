#pragma once

#include "device/context.hpp"

// Only declare class if CUDA is enabled or alternatively we can throw a compilation error

#ifdef USE_CUDA

#include <cstddef>

namespace tdevice {
class CUDAContext : public Context {
public:
  explicit CUDAContext(int id);

  size_t getTotalMemory() const override;
  size_t getAvailableMemory() const override;
  void *allocateMemory(size_t size) override;
  void deallocateMemory(void *ptr) override;
  void copyToDevice(void *dest, const void *src, size_t size) override;
  void copyToHost(void *dest, const void *src, size_t size) override;
};
} // namespace tdevice

#endif // USE_CUDA