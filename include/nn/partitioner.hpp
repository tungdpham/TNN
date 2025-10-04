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

/**
 * @brief Struct to hold throughput information for a stage/worker
 */
struct StageLoadInfo {
  float avg_forward_time_ms = 0.0f;
  float avg_backward_time_ms = 0.0f;
  float throughput_score = 1.0f;

  /**
   * @brief Computes a throughput score where higher values indicate better performance
   */
  void compute_throughput_score() {
    float total_time_ms = avg_forward_time_ms + avg_backward_time_ms;
    if (total_time_ms > 0.0f) {

      throughput_score = 1000.0f / total_time_ms;
    } else {
      throughput_score = 1.0f;
    }
  }
};

namespace LoadAwarePartitioner {

/**
 * @brief Creates balanced partitions based on computational load (FLOPs) and stage throughput
 *
 * The algorithm uses a greedy approach to assign layers to stages, attempting to balance
 * the workload/throughput ratio across all stages.
 *
 * @tparam T The data type (typically float)
 * @param model The sequential model to partition
 * @param input_shape The input shape for FLOP calculation
 * @param num_partitions Number of partitions to create
 * @param stage_load_info Vector of load information for each stage (must match num_partitions)
 * @return Vector of partitions
 */
template <typename T>
std::vector<Partition>
get_partitions(const Sequential<T> &model, const std::vector<size_t> &input_shape,
               const size_t num_partitions, const std::vector<StageLoadInfo> &stage_load_info) {

  if (num_partitions < 1) {
    throw std::invalid_argument("Number of partitions must be at least 1");
  }

  const auto &layers = model.get_layers();
  if (layers.empty()) {
    throw std::invalid_argument("Model has no layers to partition");
  }

  if (stage_load_info.size() != num_partitions) {
    throw std::invalid_argument("Stage load info size must match number of partitions");
  }

  std::vector<uint64_t> forward_flops = model.forward_complexity(input_shape);
  std::vector<uint64_t> backward_flops = model.backward_complexity(input_shape);

  if (forward_flops.size() != layers.size() || backward_flops.size() != layers.size()) {
    throw std::runtime_error("FLOP calculation mismatch with layer count");
  }

  std::vector<uint64_t> layer_flops(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    layer_flops[i] =
        static_cast<uint64_t>(forward_flops[i]) + static_cast<uint64_t>(backward_flops[i]);
  }

  uint64_t total_flops =
      std::accumulate(layer_flops.begin(), layer_flops.end(), static_cast<uint64_t>(0));

  float total_throughput = 0.0f;

  total_throughput = std::accumulate(
      stage_load_info.begin(), stage_load_info.end(), 0.0f,
      [](float sum, const StageLoadInfo &info) { return sum + info.throughput_score; });

  if (total_throughput <= 0.0f) {
    std::cerr << "Warning: Total throughput is zero or negative. Using equal distribution.\n";
    return NaivePartitioner::get_partitions(layers, num_partitions);
  }

  std::vector<uint64_t> target_flops_per_stage(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    float proportion = stage_load_info[i].throughput_score / total_throughput;
    target_flops_per_stage[i] = static_cast<uint64_t>(total_flops * proportion);
  }

  std::vector<Partition> partitions;
  std::vector<uint64_t> assigned_flops(num_partitions, 0);

  size_t current_stage = 0;
  size_t stage_start = 0;

  for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    assigned_flops[current_stage] += layer_flops[layer_idx];

    bool should_switch = false;

    if (current_stage < num_partitions - 1) {
      uint64_t current_load = assigned_flops[current_stage];
      uint64_t target_load = target_flops_per_stage[current_stage];

      if (current_load >= target_load && layer_idx < layers.size() - 1) {
        should_switch = true;

        if (layer_idx + 1 < layers.size()) {
          uint64_t next_layer_flops = layer_flops[layer_idx + 1];
          uint64_t current_diff = (current_load > target_load) ? (current_load - target_load)
                                                               : (target_load - current_load);
          uint64_t next_diff = (current_load + next_layer_flops > target_load)
                                   ? (current_load + next_layer_flops - target_load)
                                   : (target_load - current_load - next_layer_flops);

          if (next_diff < current_diff && current_load + next_layer_flops < target_load * 1.5) {
            should_switch = false;
          }
        }
      }
    }

    if (should_switch) {

      partitions.emplace_back(stage_start, layer_idx + 1);
      stage_start = layer_idx + 1;
      current_stage++;
    }
  }

  if (stage_start < layers.size()) {
    partitions.emplace_back(stage_start, layers.size());
  }

  while (partitions.size() < num_partitions) {

    size_t largest_idx = 0;
    size_t largest_size = 0;
    for (size_t i = 0; i < partitions.size(); ++i) {
      size_t size = partitions[i].end_layer - partitions[i].start_layer;
      if (size > largest_size && size > 1) {
        largest_size = size;
        largest_idx = i;
      }
    }

    if (largest_size <= 1) {
      std::cerr << "Warning: Cannot create " << num_partitions
                << " partitions with current layer count.\n";
      break;
    }

    Partition &p = partitions[largest_idx];
    size_t mid = p.start_layer + (p.end_layer - p.start_layer) / 2;
    Partition new_partition(mid, p.end_layer);
    p.end_layer = mid;
    partitions.insert(partitions.begin() + largest_idx + 1, new_partition);
  }

  while (partitions.size() > num_partitions) {

    size_t smallest_idx = 0;
    size_t smallest_size = layers.size() + 1;
    for (size_t i = 0; i < partitions.size() - 1; ++i) {
      size_t size = partitions[i].end_layer - partitions[i].start_layer;
      if (size < smallest_size) {
        smallest_size = size;
        smallest_idx = i;
      }
    }

    partitions[smallest_idx].end_layer = partitions[smallest_idx + 1].end_layer;
    partitions.erase(partitions.begin() + smallest_idx + 1);
  }

  std::cout << "\n=== Load-Aware Partitioning Summary ===\n";
  std::cout << "Total layers: " << layers.size() << "\n";
  std::cout << "Total FLOPs: " << total_flops << "\n";
  std::cout << "Number of partitions: " << partitions.size() << "\n\n";

  for (size_t i = 0; i < partitions.size(); ++i) {
    const auto &part = partitions[i];
    uint64_t partition_flops = 0;
    for (size_t j = part.start_layer; j < part.end_layer; ++j) {
      partition_flops += layer_flops[j];
    }

    float load_percentage = (total_flops > 0) ? (100.0f * partition_flops / total_flops) : 0.0f;
    float throughput_percentage =
        (total_throughput > 0) ? (100.0f * stage_load_info[i].throughput_score / total_throughput)
                               : 0.0f;

    std::cout << "Stage " << i << " (layers " << part.start_layer << "-" << part.end_layer - 1
              << "):\n";
    std::cout << "  Layers: " << (part.end_layer - part.start_layer) << "\n";
    std::cout << "  FLOPs: " << partition_flops << " (" << load_percentage << "%)\n";
    std::cout << "  Throughput score: " << stage_load_info[i].throughput_score << " ("
              << throughput_percentage << "%)\n";
    std::cout << "  Forward time: " << stage_load_info[i].avg_forward_time_ms << " ms\n";
    std::cout << "  Backward time: " << stage_load_info[i].avg_backward_time_ms << " ms\n";

    float expected_time_ratio = (stage_load_info[i].throughput_score > 0)
                                    ? (partition_flops / stage_load_info[i].throughput_score)
                                    : 0.0f;
    std::cout << "  Expected time ratio: " << expected_time_ratio << "\n\n";
  }
  std::cout << "=====================================\n\n";

  return partitions;
}

/**
 * @brief Simplified version that assumes uniform throughput across all stages
 *
 * This is useful when no throughput data is available yet, but you still want
 * to partition based on computational complexity rather than just layer count.
 *
 * @tparam T The data type (typically float)
 * @param model The sequential model to partition
 * @param input_shape The input shape for FLOP calculation
 * @param num_partitions Number of partitions to create
 * @return Vector of partitions
 */
template <typename T>
std::vector<Partition> get_partitions(const Sequential<T> &model,
                                      const std::vector<size_t> &input_shape,
                                      const size_t num_partitions) {

  std::vector<StageLoadInfo> uniform_load_info(num_partitions);
  for (auto &info : uniform_load_info) {
    info.throughput_score = 1.0f;
  }

  return get_partitions(model, input_shape, num_partitions, uniform_load_info);
}

} // namespace LoadAwarePartitioner

} // namespace tnn
