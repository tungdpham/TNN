/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/maxpool_ops.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace maxpool {
template <typename T>
__global__ void
compute_max_pool_forward_kernel(const T *input_data, T *output_data, size_t batch_size,
                                size_t channels, size_t input_h, size_t input_w, size_t output_h,
                                size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                                size_t stride_w, size_t pad_h, size_t pad_w, size_t *mask_indices) {
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

  const size_t input_offset = (n * channels + c) * input_h * input_w;

  // Calculate the theoretical window in the input (implicit padding)
  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);
  long h_end = h_start + pool_h;
  long w_end = w_start + pool_w;

  // Clamp the window to actual image boundaries
  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_end);
  long w_end_valid = min(static_cast<long>(input_w), w_end);

  T max_val = -INFINITY;
  size_t max_idx = 0;

  // Loop only over valid pixels
  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      const size_t cur_input_idx = input_offset + ih * input_w + iw;
      T val = input_data[cur_input_idx];

      if (val > max_val || (ih == h_start_valid && iw == w_start_valid)) {
        max_val = val;
        max_idx = cur_input_idx;
      }
    }
  }

  output_data[idx] = max_val;
  mask_indices[idx] = max_idx;
}

template <typename T>
__global__ void compute_max_pool_backward_kernel(const T *gradient_data, T *grad_input_data,
                                                 size_t batch_size, size_t channels,
                                                 size_t output_h, size_t output_w,
                                                 const size_t *mask_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs)
    return;

  const T grad_val = gradient_data[idx];
  const size_t input_idx = mask_indices[idx];

  atomicAdd(&grad_input_data[input_idx], grad_val);
}

template <typename T>
void compute_max_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, size_t pad_h, size_t pad_w,
                              device_ptr<size_t[]> &mask_indices, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  compute_max_pool_forward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, output_data, batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, pad_h, pad_w, mask_indices.get());
}

template <typename T>
void compute_max_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const device_ptr<size_t[]> &mask_indices, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  compute_max_pool_backward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, grad_input_data, batch_size, channels, output_h, output_w, mask_indices.get());
}

template void compute_max_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w, size_t pad_h, size_t pad_w,
                                              device_ptr<size_t[]> &mask_indices,
                                              cudaStream_t stream);
template void compute_max_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w,
                                               device_ptr<size_t[]> &mask_indices,
                                               cudaStream_t stream);

template void compute_max_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t output_h,
                                               size_t output_w,
                                               const device_ptr<size_t[]> &mask_indices,
                                               cudaStream_t stream);
template void compute_max_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t output_h, size_t output_w,
                                                const device_ptr<size_t[]> &mask_indices,
                                                cudaStream_t stream);

} // namespace maxpool
} // namespace cuda
} // namespace tnn