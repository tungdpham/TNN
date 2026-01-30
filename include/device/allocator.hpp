#pragma once

#include "device/dptr.hpp"

namespace tnn {
class IAllocator {
public:
  virtual ~IAllocator() = default;

  virtual dptr allocate(size_t size) = 0;

  virtual void deallocate(dptr &&ptr) = 0;
};
} // namespace tnn