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

// Simple naive pipeline partitioning strategy that divides layers based on given weights
class NaivePipelinePartitioner : public Partitioner {
public:
  NaivePipelinePartitioner(const NaivePartitionerConfig &config)
      : Partitioner(config.weights.size()), config_(config) {}
  ~NaivePipelinePartitioner() = default;

  std::vector<SeqPartition> partition_model(const std::vector<Layer *> &layers_) override {
    if (this->num_partitions_ == 0) {
      throw std::runtime_error("Number of partitions must be greater than zero");
    }
    std::vector<SeqPartition> partitions;
    size_t total_layers = layers_.size();

    size_t weight_sum = std::accumulate(config_.weights.begin(), config_.weights.end(), 0);

    size_t current_layer = 0;
    for (size_t i = 0; i < this->num_partitions_; ++i) {
      size_t partition_size = (total_layers * config_.weights[i]) / weight_sum;
      size_t start_layer = current_layer;
      size_t end_layer = (i == this->num_partitions_ - 1)
                             ? total_layers
                             : std::min(current_layer + partition_size, total_layers);
      size_t length = end_layer - start_layer;
      partitions.push_back(SeqPartition(start_layer, length));
      current_layer = end_layer;
    }

    return partitions;
  }

  std::vector<InputPartition> partition_input(const Tensor &input, const Tensor &labels) override {
    if (!input || input->shape().empty()) {
      throw std::runtime_error("Input tensor is null or has empty shape");
    }
    size_t batch_size = input->dimension(0);
    // same config for every stage
    std::vector<InputPartition> input_partitions;
    input_partitions.push_back(InputPartition(0, batch_size));
    return input_partitions;
  }

private:
  NaivePartitionerConfig config_;
};

// Simple naive data partitioning strategy that divides input tensor based on weights
class NaiveDataPartitioner : public Partitioner {
public:
  NaiveDataPartitioner(const NaivePartitionerConfig &config)
      : Partitioner(config.weights.size()), config_(config) {}
  ~NaiveDataPartitioner() = default;

  std::vector<SeqPartition> partition_model(const std::vector<Layer *> &layers) override {
    // same config for every stage
    std::vector<SeqPartition> partitions;
    partitions.push_back(SeqPartition(0, layers.size()));
    return partitions;
  }

  std::vector<InputPartition> partition_input(const Tensor &input, const Tensor &labels) override {
    if (this->num_partitions_ == 0) {
      throw std::runtime_error("Number of partitions must be greater than zero");
    }
    if (!input || input->shape().empty()) {
      throw std::runtime_error("Input tensor is null or has empty shape");
    }

    std::vector<InputPartition> input_partitions;
    size_t batch_size = input->dimension(0);
    size_t weight_sum = std::accumulate(config_.weights.begin(), config_.weights.end(), 0);

    size_t current_index = 0;
    for (size_t i = 0; i < this->num_partitions_; ++i) {
      size_t partition_size = (batch_size * config_.weights[i]) / weight_sum;
      size_t start_index = current_index;
      size_t end_index = (i == this->num_partitions_ - 1)
                             ? batch_size
                             : std::min(current_index + partition_size, batch_size);
      size_t length = end_index - start_index;
      input_partitions.push_back(InputPartition(start_index, length));
      current_index = end_index;
    }

    return input_partitions;
  }

private:
  NaivePartitionerConfig config_;
};
}  // namespace tnn