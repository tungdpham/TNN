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

template <typename T>
__global__ void compute_avg_pool_forward_kernel(const T *input_data, T *output_data,
                                                size_t batch_size, size_t channels, size_t input_h,
                                                size_t input_w, size_t output_h, size_t output_w,
                                                size_t pool_h, size_t pool_w, size_t stride_h,
                                                size_t stride_w, T pool_size_inv) {
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

  T sum = T(0);
  for (size_t ph = 0; ph < pool_h; ++ph) {
    for (size_t pw = 0; pw < pool_w; ++pw) {
      const size_t h_idx = out_h * stride_h + ph;
      const size_t w_idx = out_w * stride_w + pw;

      const size_t input_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
      sum += input_data[input_idx];
    }
  }

  output_data[idx] = sum * pool_size_inv;
}

template <typename T>
__global__ void compute_avg_pool_backward_kernel(const T *gradient_data, T *grad_input_data,
                                                 size_t batch_size, size_t channels, size_t input_h,
                                                 size_t input_w, size_t output_h, size_t output_w,
                                                 size_t pool_h, size_t pool_w, size_t stride_h,
                                                 size_t stride_w, T pool_size_inv) {
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

  const T grad_val = gradient_data[idx] * pool_size_inv;

  for (size_t ph = 0; ph < pool_h; ++ph) {
    for (size_t pw = 0; pw < pool_w; ++pw) {
      const size_t h_idx = out_h * stride_h + ph;
      const size_t w_idx = out_w * stride_w + pw;

      const size_t input_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
      atomicAdd(&grad_input_data[input_idx], grad_val);
    }
  }
}

template <typename T>
void compute_avg_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  compute_avg_pool_forward_kernel<<<num_blocks, threads_per_block>>>(
      input_data, output_data, batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, pool_size_inv);

  cudaDeviceSynchronize();
}

template <typename T>
void compute_avg_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  compute_avg_pool_backward_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, grad_input_data, batch_size, channels, input_h, input_w, output_h, output_w,
      pool_h, pool_w, stride_h, stride_w, pool_size_inv);

  cudaDeviceSynchronize();
}

// Explicit template instantiations
template void compute_avg_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w);
template void compute_avg_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w);

template void compute_avg_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w);
template void compute_avg_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t input_h, size_t input_w,
                                                size_t output_h, size_t output_w, size_t pool_h,
                                                size_t pool_w, size_t stride_h, size_t stride_w);

} // namespace cuda
} // namespace tnn
