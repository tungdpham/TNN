#pragma once

#include <cstdint>

#include "device/device.hpp"
#include "device/device_manager.hpp"

namespace tnn {

template <typename T>
struct Blob {
  using value_type = T;
  T* ptr;
  uint64_t count;
  const Device& device;
};

template <typename T>
  requires(!std::is_same_v<T, void>)
Blob<T> make_blob(T* data, uint64_t count, const Device& device = getHost()) {
  return Blob<T>{data, count, device};
}

}  // namespace tnn