/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "utils/hardware_info.hpp"
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

struct LoadTracker {
  /**
   * Performance metrics to send to coordinator
   * These are serialized and sent periodically.
   */
  // throughputs
  uint32_t avg_forward_time_ = 0;  // avg forward time per batch in milliseconds
  uint32_t avg_backward_time_ = 0; // avg backward time per batch in milliseconds

  // system metrics
  float avg_cpu_utilization_ = -1.0f; // CPU utilization ratio (0.0 to 1.0)
  float max_memory_usage_ = -1.0f;    // Maximum memory usage in MB, -1.0 if unavailable

  static std::vector<uint8_t> serialize(const LoadTracker &tracker) {
    std::vector<uint8_t> buffer;
    uint8_t *buffer_data = buffer.data();

    buffer.resize(sizeof(LoadTracker));
    size_t offset = 0;
    std::memcpy(buffer_data + offset, &tracker.avg_forward_time_, sizeof(uint32_t));

    offset += sizeof(uint32_t);
    std::memcpy(buffer_data + offset, &tracker.avg_backward_time_, sizeof(uint32_t));

    offset += sizeof(uint32_t);
    std::memcpy(buffer_data + offset, &tracker.avg_cpu_utilization_, sizeof(float));

    offset += sizeof(float);
    std::memcpy(buffer_data + offset, &tracker.max_memory_usage_, sizeof(float));

    return buffer;
  }

  static LoadTracker deserialize(const std::vector<uint8_t> &buffer) {
    LoadTracker tracker;
    const uint8_t *buffer_data = buffer.data();

    if (buffer.size() < sizeof(LoadTracker)) {
      throw std::runtime_error("Invalid buffer size for LoadTracker deserialization");
    }

    size_t offset = 0;
    std::memcpy(&tracker.avg_forward_time_, buffer_data + offset, sizeof(uint32_t));

    offset += sizeof(uint32_t);
    std::memcpy(&tracker.avg_backward_time_, buffer_data + offset, sizeof(uint32_t));

    offset += sizeof(uint32_t);
    std::memcpy(&tracker.avg_cpu_utilization_, buffer_data + offset, sizeof(float));

    offset += sizeof(float);
    std::memcpy(&tracker.max_memory_usage_, buffer_data + offset, sizeof(float));

    return tracker;
  }
};