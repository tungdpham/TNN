/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/partitioner.hpp"
#include "nn/sequential.hpp"
#include "pipeline/load_tracker.hpp"
#include <iostream>
#include <vector>

/**
 * @brief Example demonstrating the Load-Aware Partitioner
 *
 * This example shows how to use the LoadAwarePartitioner to create balanced
 * partitions based on:
 * 1. Computational complexity (FLOPs) of each layer
 * 2. Throughput capabilities of each stage/worker
 */

int main() {
  using namespace tnn;

  std::cout << "=== Load-Aware Partitioner Example ===\n\n";

  // Create a sample CNN model similar to CIFAR-10 classifier
  auto model = SequentialBuilder<float>("example_cnn")
                   .input({3, 32, 32})
                   .conv2d(32, 3, 3, 1, 1, 1, 1, true, "conv1")
                   .activation("relu", "relu1")
                   .conv2d(32, 3, 3, 1, 1, 1, 1, true, "conv2")
                   .activation("relu", "relu2")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv3")
                   .activation("relu", "relu3")
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv4")
                   .activation("relu", "relu4")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .flatten("flatten")
                   .dense(512, "relu", true, "fc1")
                   .dense(10, "none", true, "fc2")
                   .build();

  std::cout << "Model created with " << model.layer_size() << " layers\n\n";

  // Define input shape (batch_size=1, channels=3, height=32, width=32)
  std::vector<size_t> input_shape = {1, 3, 32, 32};

  // Number of stages/partitions
  size_t num_stages = 3;

  std::cout << "Example 1: Uniform throughput (no load data)\n";
  std::cout << "----------------------------------------------\n";

  // Use the simplified version without load data
  auto uniform_partitions = LoadAwarePartitioner::get_partitions(model, input_shape, num_stages);

  std::cout << "\nExample 2: With simulated load tracker data\n";
  std::cout << "----------------------------------------------\n";

  // Simulate load tracker data for 3 stages
  // Stage 0: Fast worker (high throughput)
  // Stage 1: Medium worker (medium throughput)
  // Stage 2: Slow worker (low throughput)
  std::vector<StageLoadInfo> stage_load_info(num_stages);

  // Stage 0: Fast worker
  stage_load_info[0].avg_forward_time_ms = 10.0f;
  stage_load_info[0].avg_backward_time_ms = 15.0f;
  stage_load_info[0].compute_throughput_score();

  // Stage 1: Medium worker
  stage_load_info[1].avg_forward_time_ms = 20.0f;
  stage_load_info[1].avg_backward_time_ms = 30.0f;
  stage_load_info[1].compute_throughput_score();

  // Stage 2: Slow worker
  stage_load_info[2].avg_forward_time_ms = 30.0f;
  stage_load_info[2].avg_backward_time_ms = 45.0f;
  stage_load_info[2].compute_throughput_score();

  std::cout << "Stage load information:\n";
  for (size_t i = 0; i < num_stages; ++i) {
    std::cout << "  Stage " << i << ": forward=" << stage_load_info[i].avg_forward_time_ms
              << "ms, backward=" << stage_load_info[i].avg_backward_time_ms
              << "ms, throughput_score=" << stage_load_info[i].throughput_score << "\n";
  }
  std::cout << "\n";

  // Create partitions with load awareness
  auto load_aware_partitions =
      LoadAwarePartitioner::get_partitions(model, input_shape, num_stages, stage_load_info);

  std::cout << "\nExample 3: Comparison with naive partitioning\n";

  const auto &layers = model.get_layers();
  auto naive_partitions = NaivePartitioner::get_partitions(layers, num_stages);

  std::cout << "Naive partitioning (equal layer count):\n";
  for (size_t i = 0; i < naive_partitions.size(); ++i) {
    const auto &part = naive_partitions[i];
    std::cout << "  Stage " << i << ": layers " << part.start_layer << "-" << part.end_layer - 1
              << " (" << (part.end_layer - part.start_layer) << " layers)\n";
  }
  std::cout << "\n";

  std::cout << "\nExample 4: Using LoadTracker struct\n";

  std::vector<LoadTracker> load_trackers(num_stages);
  load_trackers[0].avg_forward_time_ = 12.0f;
  load_trackers[0].avg_backward_time_ = 18.0f;
  load_trackers[1].avg_forward_time_ = 25.0f;
  load_trackers[1].avg_backward_time_ = 35.0f;
  load_trackers[2].avg_forward_time_ = 40.0f;
  load_trackers[2].avg_backward_time_ = 60.0f;

  // Convert LoadTracker to StageLoadInfo
  std::vector<StageLoadInfo> converted_load_info(num_stages);
  for (size_t i = 0; i < num_stages; ++i) {
    converted_load_info[i].avg_forward_time_ms = load_trackers[i].avg_forward_time_;
    converted_load_info[i].avg_backward_time_ms = load_trackers[i].avg_backward_time_;
    converted_load_info[i].compute_throughput_score();
  }

  auto tracker_partitions =
      LoadAwarePartitioner::get_partitions(model, input_shape, num_stages, converted_load_info);

  std::cout << "=== Example Complete ===\n";

  return 0;
}
