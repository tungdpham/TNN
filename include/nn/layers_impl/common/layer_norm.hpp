/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {

struct LayerNormStats {
  size_t batch_size = 0;
  size_t channels = 0;
  bool affine = false;
  float epsilon = 1e-5f;
  size_t fwd_workspace_size = 0;
  size_t bwd_workspace_size = 0;
};

inline void init_layer_norm_stats(LayerNormStats &stats, size_t batch_size, size_t channels,
                                  bool affine, float epsilon) {
  stats.batch_size = batch_size;
  stats.channels = channels;
  stats.affine = affine;
  stats.epsilon = epsilon;
  stats.fwd_workspace_size = 0;
  stats.bwd_workspace_size = 0;
}

inline void round_workspace_size(LayerNormStats &stats, size_t alignment = 16) {
  stats.fwd_workspace_size = ((stats.fwd_workspace_size + alignment - 1) / alignment) * alignment;
  stats.bwd_workspace_size = ((stats.bwd_workspace_size + alignment - 1) / alignment) * alignment;
}

}  // namespace tnn
