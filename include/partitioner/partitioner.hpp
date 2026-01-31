#pragma once

#include "nn/sequential.hpp"

namespace tnn {
class Partitioner {
public:
  Partitioner() = default;
  virtual ~Partitioner() = default;

  virtual std::vector<Partition> get_partitions(const std::vector<Layer *> &layers) = 0;
};

}  // namespace tnn