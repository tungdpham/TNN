#pragma once

#include <memory>

struct LoadTracker { 
  float avg_cpu_utilization = -1.0f;   // CPU utilization ratio of this process (0.0 to 1.0)
  float max_memory_usage = -1.0f; // Maximum memory usage in MB, -1.0 if unavailable
  uint32_t avg_forward_time = 0; // avg forward time per batch in milliseconds
  uint32_t avg_backward_time = 0; // avg backward time per batch in milliseconds
  uint32_t batches_processed = 0; // total number of batches processed
};