/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {

struct AttentionStats {
  size_t batch_size = 0;
  size_t num_heads = 0;
  size_t seq_len = 0;
  size_t head_dim = 0;
  float attn_scale = 1.0f;
  bool is_causal = false;
  size_t fwd_workspace_size = 0;
  size_t bwd_workspace_size = 0;
};

inline void init_attention_stats(AttentionStats &stats, size_t batch_size, size_t num_heads,
                                 size_t seq_len, size_t head_dim, float attn_scale,
                                 bool is_causal) {
  stats.batch_size = batch_size;
  stats.num_heads = num_heads;
  stats.seq_len = seq_len;
  stats.head_dim = head_dim;
  stats.attn_scale = attn_scale;
  stats.is_causal = is_causal;
  stats.fwd_workspace_size = 0;
  stats.bwd_workspace_size = 0;
}

inline void round_attention_workspace_size(AttentionStats &stats, size_t alignment = 16) {
  stats.fwd_workspace_size = ((stats.fwd_workspace_size + alignment - 1) / alignment) * alignment;
  stats.bwd_workspace_size = ((stats.bwd_workspace_size + alignment - 1) / alignment) * alignment;
}

}  // namespace tnn
