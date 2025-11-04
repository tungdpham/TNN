#include "device/contexts/cpu_context.hpp"

#include <cstdlib>
#include <cstring>

namespace tdevice {

void *CPUContext::allocateMemory(size_t size) { return std::malloc(size); }

void CPUContext::deallocateMemory(void *ptr) { std::free(ptr); }

void CPUContext::copyToDevice(void *dest, const void *src, size_t size) {
  std::memcpy(dest, src, size);
}

void CPUContext::copyToHost(void *dest, const void *src, size_t size) {
  std::memcpy(dest, src, size);
}

} // namespace tdevice