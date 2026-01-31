#pragma once

#include <vector>

#include "partitioner.hpp"

namespace tnn {

struct NaivePartitionerConfig {
  std::vector<size_t> weights;  // weight for each partition

  NaivePartitionerConfig(const std::vector<size_t> &weights) {
    assert(!weights.empty() && "Weights vector must not be empty");
    this->weights = weights;
  }
};

class NaivePartitioner : public Partitioner {
  // Simple naive partitioning strategy that divides layers evenly among partitions
public:
  NaivePartitioner(const NaivePartitionerConfig &config) : config_(config) {}
  ~NaivePartitioner() = default;

  std::vector<Partition> get_partitions(const std::vector<Layer *> &layers_) override {
    size_t num_partitions = config_.weights.size();
    if (num_partitions < 1) {
      throw std::invalid_argument("Number of partitions must be at least 1");
    }
    std::vector<Partition> partitions;
    size_t total_layers = layers_.size();

    size_t weight_sum = std::accumulate(config_.weights.begin(), config_.weights.end(), 0);

    size_t current_layer = 0;
    for (size_t i = 0; i < num_partitions; ++i) {
      size_t partition_size = (total_layers * config_.weights[i]) / weight_sum;
      size_t start_layer = current_layer;
      size_t end_layer = (i == num_partitions - 1)
                             ? total_layers
                             : std::min(current_layer + partition_size, total_layers);
      partitions.emplace_back(start_layer, end_layer);
      current_layer = end_layer;
    }

    return partitions;
  }

private:
  NaivePartitionerConfig config_;
};

}  // namespace tnn