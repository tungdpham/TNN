/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
struct BatchNormStats {
  size_t batch_size = 0;
  size_t channels = 0;
  size_t height = 0;
  size_t width = 0;
  double epsilon = 1e-5;
  double momentum = 0.1;
  bool use_relu = false;
  size_t fwd_workspace_size = 0;
  size_t bwd_workspace_size = 0;
  size_t inf_workspace_size = 0;
};

inline void init_batchnorm_stats(BatchNormStats &stats, size_t batch_size, size_t channels,
                                 size_t height, size_t width, double epsilon, double momentum,
                                 bool use_relu) {
  stats.batch_size = batch_size;
  stats.channels = channels;
  stats.height = height;
  stats.width = width;
  stats.epsilon = epsilon;
  stats.momentum = momentum;
  stats.use_relu = use_relu;
  stats.fwd_workspace_size = 0;
  stats.bwd_workspace_size = 0;
  stats.inf_workspace_size = 0;
}

inline void round_workspace_size(BatchNormStats &stats, size_t alignment = 16) {
  stats.fwd_workspace_size = ((stats.fwd_workspace_size + alignment - 1) / alignment) * alignment;
  stats.bwd_workspace_size = ((stats.bwd_workspace_size + alignment - 1) / alignment) * alignment;
  stats.inf_workspace_size = ((stats.inf_workspace_size + alignment - 1) / alignment) * alignment;
}

}  // namespace tnn