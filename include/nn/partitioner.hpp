#pragma once

#include "sequential.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tnn {

namespace NaivePartitioner {
template <typename T>
std::vector<Partition> get_partitions(const std::vector<std::unique_ptr<Layer<T>>> &layers_,
                                      const size_t num_partitions) {
  if (num_partitions < 1) {
    throw std::invalid_argument("Number of partitions must be at least 1");
  }
  std::vector<Partition> partitions;
  size_t total_layers = layers_.size();
  size_t base_partition_size = total_layers / num_partitions;
  size_t remainder = total_layers % num_partitions;

  size_t current_start = 0;
  for (size_t i = 0; i < num_partitions; ++i) {
    size_t current_partition_size = base_partition_size + (i < remainder ? 1 : 0);
    size_t current_end = current_start + current_partition_size;

    partitions.emplace_back(current_start, current_end);
    current_start = current_end;
  }

  return partitions;
}
} // namespace NaivePartitioner

} // namespace tnn
