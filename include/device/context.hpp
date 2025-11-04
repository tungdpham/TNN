#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace tdevice {
class Context {
public:
  Context(int id) : id_(id) {}
  ~Context() = default;

  int getID() const { return id_; }

  virtual size_t getTotalMemory() const = 0;
  virtual size_t getAvailableMemory() const = 0;
  virtual void *allocateMemory(size_t size) = 0;
  virtual void deallocateMemory(void *ptr) = 0;
  virtual void copyToDevice(void *dest, const void *src, size_t size) = 0;
  virtual void copyToHost(void *dest, const void *src, size_t size) = 0;

private:
  int id_;
};
} // namespace tdevice
