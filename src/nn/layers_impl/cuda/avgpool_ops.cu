/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/avgpool_ops.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace avgpool {
template <typename T>
__global__ void
compute_avg_pool_forward_kernel(const T *input_data, T *output_data, size_t batch_size,
                                size_t channels, size_t input_h, size_t input_w, size_t output_h,
                                size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                                size_t stride_w, size_t pad_h, size_t pad_w, T pool_size_inv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs)
    return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  // Calculate virtual window position
  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

  // Clamp to valid image area
  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
  long w_end_valid = min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

  const size_t input_offset = (n * channels + c) * input_h * input_w;
  T sum = T(0);

  // Accumulate only valid pixels
  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      sum += input_data[input_offset + ih * input_w + iw];
    }
  }

  output_data[idx] = sum * pool_size_inv;
}

template <typename T>
__global__ void
compute_avg_pool_backward_kernel(const T *gradient_data, T *grad_input_data, size_t batch_size,
                                 size_t channels, size_t input_h, size_t input_w, size_t output_h,
                                 size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                                 size_t stride_w, size_t pad_h, size_t pad_w, T pool_size_inv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs)
    return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  // Gradient distributed equally to all contributing pixels
  const T grad_val = gradient_data[idx] * pool_size_inv;

  // Same boundary logic as forward pass
  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
  long w_end_valid = min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

  const size_t input_offset = (n * channels + c) * input_h * input_w;

  // Scatter add
  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      atomicAdd(&grad_input_data[input_offset + ih * input_w + iw], grad_val);
    }
  }
}

template <typename T>
void compute_avg_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  compute_avg_pool_forward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, output_data, batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, pad_h, pad_w, pool_size_inv);
}

template <typename T>
void compute_avg_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  compute_avg_pool_backward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, grad_input_data, batch_size, channels, input_h, input_w, output_h, output_w,
      pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, pool_size_inv);
}

template void compute_avg_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w, size_t pad_h, size_t pad_w,
                                              cudaStream_t stream);
template void compute_avg_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w,
                                               cudaStream_t stream);

template void compute_avg_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w,
                                               cudaStream_t stream);
template void compute_avg_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t input_h, size_t input_w,
                                                size_t output_h, size_t output_w, size_t pool_h,
                                                size_t pool_w, size_t stride_h, size_t stride_w,
                                                size_t pad_h, size_t pad_w, cudaStream_t stream);
} // namespace avgpool
} // namespace cuda
} // namespace tnn
