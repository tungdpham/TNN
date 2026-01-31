/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {

struct ConvolutionStats {
  size_t batch_size = 0;
  size_t in_channels = 0;
  size_t input_h = 0;
  size_t input_w = 0;
  size_t out_channels = 0;
  size_t output_h = 0;
  size_t output_w = 0;
  size_t kernel_h = 0;
  size_t kernel_w = 0;
  size_t stride_h = 0;
  size_t stride_w = 0;
  size_t pad_h = 0;
  size_t pad_w = 0;
  bool use_bias = false;
  size_t fwd_workspace_size = 0;
  size_t dgrad_workspace_size = 0;
  size_t wgrad_workspace_size = 0;
  size_t bgrad_workspace_size = 0;
};

inline void init_convolution_stats(ConvolutionStats &stats, size_t batch_size, size_t in_channels,
                                   size_t input_h, size_t input_w, size_t out_channels,
                                   size_t kernel_h, size_t kernel_w, size_t stride_h,
                                   size_t stride_w, size_t pad_h, size_t pad_w, bool use_bias) {
  stats.batch_size = batch_size;
  stats.in_channels = in_channels;
  stats.input_h = input_h;
  stats.input_w = input_w;
  stats.out_channels = out_channels;
  stats.kernel_h = kernel_h;
  stats.kernel_w = kernel_w;
  stats.stride_h = stride_h;
  stats.stride_w = stride_w;
  stats.pad_h = pad_h;
  stats.pad_w = pad_w;
  stats.use_bias = use_bias;

  // Calculate output dimensions
  stats.output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
  stats.output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

  stats.fwd_workspace_size = 0;
  stats.dgrad_workspace_size = 0;
  stats.wgrad_workspace_size = 0;
  stats.bgrad_workspace_size = 0;
}

inline void round_workspace_size(ConvolutionStats &stats, size_t alignment = 16) {
  stats.fwd_workspace_size = ((stats.fwd_workspace_size + alignment - 1) / alignment) * alignment;
  stats.dgrad_workspace_size =
      ((stats.dgrad_workspace_size + alignment - 1) / alignment) * alignment;
  stats.wgrad_workspace_size =
      ((stats.wgrad_workspace_size + alignment - 1) / alignment) * alignment;
  stats.bgrad_workspace_size =
      ((stats.bgrad_workspace_size + alignment - 1) / alignment) * alignment;
}

}  // namespace tnn
