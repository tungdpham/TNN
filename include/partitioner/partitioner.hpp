#pragma once

#include "nn/sequential.hpp"

namespace tnn {
template <typename T> class Partitioner {
public:
  Partitioner() = default;
  virtual ~Partitioner() = default;

  virtual std::vector<Partition> get_partitions(const std::vector<Layer<T> *> &layers) = 0;
};

} // namespace tnn