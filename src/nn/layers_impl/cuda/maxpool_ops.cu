/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/maxpool_ops.hpp"

#include <cuda_runtime.h>
#include <limits>

namespace tnn {
namespace cuda {
namespace maxpool {
template <typename T>
__global__ void compute_max_pool_forward_kernel(const T *input_data, T *output_data,
                                                size_t batch_size, size_t channels, size_t input_h,
                                                size_t input_w, size_t output_h, size_t output_w,
                                                size_t pool_h, size_t pool_w, size_t stride_h,
                                                size_t stride_w, size_t *mask_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs)
    return;

  // Decode indices
  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  // Initialize with the first value from the pooling window
  const size_t first_h_idx = out_h * stride_h;
  const size_t first_w_idx = out_w * stride_w;
  const size_t first_target_idx =
      ((n * channels + c) * input_h + first_h_idx) * input_w + first_w_idx;
  T max_val = input_data[first_target_idx];
  size_t max_idx = first_target_idx;

  for (size_t ph = 0; ph < pool_h; ++ph) {
    for (size_t pw = 0; pw < pool_w; ++pw) {
      const size_t h_idx = out_h * stride_h + ph;
      const size_t w_idx = out_w * stride_w + pw;

      const size_t target_padded_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
      T val = input_data[target_padded_idx];
      if (val > max_val) {
        max_val = val;
        max_idx = target_padded_idx;
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
                              size_t stride_w, std::vector<size_t> &mask_indices) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  // Allocate GPU memory for mask indices
  size_t *d_mask_indices;
  cudaMalloc(&d_mask_indices, mask_indices.size() * sizeof(size_t));

  compute_max_pool_forward_kernel<<<num_blocks, threads_per_block>>>(
      input_data, output_data, batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, d_mask_indices);

  cudaDeviceSynchronize();

  // Copy mask indices back to host
  cudaMemcpy(mask_indices.data(), d_mask_indices, mask_indices.size() * sizeof(size_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_mask_indices);
}

template <typename T>
void compute_max_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const std::vector<size_t> &mask_indices) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  // Allocate and copy mask indices to GPU
  size_t *d_mask_indices;
  cudaMalloc(&d_mask_indices, mask_indices.size() * sizeof(size_t));
  cudaMemcpy(d_mask_indices, mask_indices.data(), mask_indices.size() * sizeof(size_t),
             cudaMemcpyHostToDevice);

  compute_max_pool_backward_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, grad_input_data, batch_size, channels, output_h, output_w, d_mask_indices);

  cudaDeviceSynchronize();

  cudaFree(d_mask_indices);
}

// Explicit template instantiations
template void compute_max_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w, std::vector<size_t> &mask_indices);
template void compute_max_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w, std::vector<size_t> &mask_indices);

template void compute_max_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t output_h,
                                               size_t output_w,
                                               const std::vector<size_t> &mask_indices);
template void compute_max_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t output_h, size_t output_w,
                                                const std::vector<size_t> &mask_indices);

} // namespace maxpool
} // namespace cuda
} // namespace tnn