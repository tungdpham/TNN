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

  /**
   * Temporary attributes to help calculate averages
   * DO NOT SERIALIZE THESE
   */

  /**
   * Helper functions
   */
  void reset() {
    avg_forward_time_ = 0;
    avg_backward_time_ = 0;

    avg_cpu_utilization_ = -1.0f;
    max_memory_usage_ = -1.0f;
  }

  LoadTracker() = default;

  ~LoadTracker() = default;
};