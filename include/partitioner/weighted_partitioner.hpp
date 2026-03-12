#pragma once

#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "nn/layer.hpp"
#include "partitioner/partitioner.hpp"

namespace tnn {
struct WeightedPartitionerConfig {
  // weight (computing power) for each workers.
  std::vector<int64_t> weights;
  std::vector<size_t> in_shape;

  WeightedPartitionerConfig(const std::vector<int64_t> &worker_weights,
                            const std::vector<size_t> &input_shape)
      : weights(worker_weights),
        in_shape(input_shape) {}
};

// // Partitions a sequence of layers based on FLOPs
// class WeightedPipelinePartitioner : public Partitioner {
// public:
//   WeightedPipelinePartitioner(WeightedPartitionerConfig config)
//       : Partitioner(config.weights.size()),
//         config_(config) {}

//   std::vector<SeqPartition> partition_model(const std::vector<Layer *> &layers) override {
//     if (this->num_partitions_ == 0) {
//       throw std::runtime_error("Number of partitions must be greater than zero");
//     }
//     if (layers.empty()) {
//       throw std::runtime_error("Cannot partition an empty model");
//     }
//     std::vector<size_t> &in_shape = config_.in_shape;
//     std::vector<SeqPartition> partitions;
//     std::vector<uint64_t> flops_pre(
//         layers.size() + 1,
//         0);  // index is shifted right by 1. 0th index is reserved for no layer

//     for (size_t i = 0; i < layers.size(); ++i) {
//       flops_pre[i + 1] =
//           flops_pre[i] + layers[i]->forward_flops(in_shape) +
//           layers[i]->backward_flops(in_shape);
//       in_shape = layers[i]->compute_output_shape(in_shape);
//     }

//     uint64_t weight_sum = 0;
//     for (const auto &weight : config_.weights) {
//       weight_sum += weight;
//     }

//     size_t current_layer = 0;
//     uint64_t total_flops = flops_pre.back();
//     for (size_t i = 0; i < this->num_partitions_; ++i) {
//       uint64_t target_flops = (total_flops * config_.weights[i]) / weight_sum;
//       size_t start_layer = current_layer;
//       auto it = std::upper_bound(flops_pre.begin() + current_layer + 1, flops_pre.end(),
//                                  flops_pre[current_layer] + target_flops);
//       size_t end_layer = it - flops_pre.begin();
//       end_layer = std::min(
//           std::max(end_layer, start_layer + 1),
//           layers.size());  // ensure at least one layer per partition but not exceed total layers
//       size_t length = end_layer - start_layer;
//       if (length == 0) {
//         throw std::runtime_error("WeightedPartitioner failed");
//       }
//       partitions.push_back(SeqPartition(start_layer, length));
//       current_layer = end_layer;
//     }

//     return partitions;
//   }

//   std::vector<InputPartition> partition_input(const ConstTensor &input,
//                                               const ConstTensor &labels) override {
//     if (!input || input->shape().empty()) {
//       throw std::runtime_error("Input tensor is null or has empty shape");
//     }
//     size_t batch_size = input->dimension(0);
//     // same config for every stage
//     std::vector<InputPartition> input_partitions;
//     input_partitions.push_back(InputPartition(0, batch_size));
//     return input_partitions;
//   }

// private:
//   WeightedPartitionerConfig config_;
// };
}  // namespace tnn