#pragma once

#include "device/dptr.hpp"

namespace tnn {

// Allocators should return a dptr whose internal storage is reclaimed by the allocator itself
class IAllocator {
public:
  virtual ~IAllocator() = default;

  // allocate a ptr with capacity >= size
  virtual dptr allocate(size_t size) = 0;
};
}  // namespace tnn