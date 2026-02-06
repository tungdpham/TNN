#pragma once

#include <cstdint>

#include "partitioner/partitioner.hpp"

namespace tnn {

struct FTDPartitionerConfig {
  std::vector<uint64_t> compute_powers;  // compute power for each worker
  uint64_t bandwidth;                    // network bandwidth between workers
  uint64_t network_delay;                // network delay for the whole setup
  std::vector<size_t> in_shape;

  FTDPartitionerConfig(const std::vector<uint64_t> &worker_compute_powers,
                       const uint64_t &network_bandwidth, uint64_t net_delay,
                       const std::vector<size_t> &input_shape)
      : compute_powers(worker_compute_powers),
        bandwidth(network_bandwidth),
        network_delay(net_delay),
        in_shape(input_shape) {}
};

// A more advanced partitioner that partitions based on workers' compute power and network
// bandwidth. Stands for Flops, Transfer, and Delay
// Config: a list of worker compute power, network bandwidth, and network delay (for the whole setup
// since most setup uses homogeneous network) It considers the total FLOPs of each layer and the
// cost of transferring intermediate activations between the partitions
class FTDPartitioner : public Partitioner {
public:
  FTDPartitioner(FTDPartitionerConfig config)
      : Partitioner(config.compute_powers.size()),
        config_(config) {}

  ~FTDPartitioner() {}

  std::vector<SeqPartition> partition_model(const std::vector<Layer *> &layers) override;

  std::vector<InputPartition> partition_input(const ConstTensor &input,
                                              const ConstTensor &labels) override {
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
  FTDPartitionerConfig config_;
};

}  // namespace tnn