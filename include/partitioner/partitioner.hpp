#pragma once

#include "nn/sequential.hpp"

namespace partitioner {
template <typename T> class Partitioner {
public:
  Partitioner() = default;
  virtual ~Partitioner() = default;

  virtual std::vector<tnn::Partition>
  get_partitions(const std::vector<std::unique_ptr<tnn::Layer<T>>> &layers,
                 const size_t num_partitions) = 0;
};

} // namespace partitioner