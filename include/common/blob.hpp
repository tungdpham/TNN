#pragma once

#include <cstdint>

namespace tnn {

template <typename T>
struct Blob {
  using value_type = T;
  T* ptr;
  uint64_t count;
};

template <typename T>
Blob<T> blob(T* data, uint64_t count) {
  return Blob<T>{data, count};
}

}  // namespace tnn