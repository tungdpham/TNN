/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {

struct MaxPoolStats {
  size_t batch_size = 0;
  size_t channels = 0;
  size_t input_h = 0;
  size_t input_w = 0;
  size_t output_h = 0;
  size_t output_w = 0;
  size_t pool_h = 0;
  size_t pool_w = 0;
  size_t stride_h = 0;
  size_t stride_w = 0;
  size_t pad_h = 0;
  size_t pad_w = 0;
  size_t fwd_workspace_size = 0;   // DNNL scratchpad for forward_training
  size_t inf_workspace_size = 0;   // DNNL scratchpad for forward_inference
  size_t bwd_workspace_size = 0;   // DNNL scratchpad for backward
  size_t pool_workspace_size = 0;  // DNNL internal index buffer (flows fwd -> bwd)
};

inline void init_maxpool_stats(MaxPoolStats &stats, size_t batch_size, size_t input_h,
                               size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                               size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
  stats.batch_size = batch_size;
  stats.channels = channels;
  stats.input_h = input_h;
  stats.input_w = input_w;
  stats.output_h = (input_h + 2 * pad_h - pool_h) / stride_h + 1;
  stats.output_w = (input_w + 2 * pad_w - pool_w) / stride_w + 1;
  stats.pool_h = pool_h;
  stats.pool_w = pool_w;
  stats.stride_h = stride_h;
  stats.stride_w = stride_w;
  stats.pad_h = pad_h;
  stats.pad_w = pad_w;
  stats.fwd_workspace_size = 0;
  stats.inf_workspace_size = 0;
  stats.bwd_workspace_size = 0;
  stats.pool_workspace_size = 0;
}

inline void round_workspace_size(MaxPoolStats &stats, size_t alignment = 16) {
  stats.fwd_workspace_size = ((stats.fwd_workspace_size + alignment - 1) / alignment) * alignment;
  stats.inf_workspace_size = ((stats.inf_workspace_size + alignment - 1) / alignment) * alignment;
  stats.bwd_workspace_size = ((stats.bwd_workspace_size + alignment - 1) / alignment) * alignment;
  // Note: pool_workspace_size is an opaque DNNL buffer; do not round-up here.
}

}  // namespace tnn
