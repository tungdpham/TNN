/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/batchnorm_ops.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace batchnorm {
template <typename T>
__global__ void compute_channel_mean_kernel(const T *input_data, T *mean_data, size_t batch_size,
                                            size_t channels, size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T sum = T(0);
  for (size_t n = 0; n < batch_size; ++n) {
    const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      sum += batch_channel_ptr[i];
    }
  }

  mean_data[c] = sum * inv_total;
}

template <typename T>
__global__ void compute_channel_variance_kernel(const T *input_data, const T *mean_data,
                                                T *var_data, size_t batch_size, size_t channels,
                                                size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);
  const T mean_val = mean_data[c];
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T sum_sq = T(0);
  for (size_t n = 0; n < batch_size; ++n) {
    const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      T diff = batch_channel_ptr[i] - mean_val;
      sum_sq += diff * diff;
    }
  }

  var_data[c] = sum_sq * inv_total;
}

template <typename T>
__global__ void normalize_and_scale_kernel(const T *input_data, const T *mean_data,
                                           const T *std_data, const T *gamma_data,
                                           const T *beta_data, T *output_data, T *normalized_data,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  const T mean_val = mean_data[c];
  const T std_val = std_data[c];
  const T inv_std = T(1) / std_val;

  T input_val = input_data[idx];
  T normalized_val = (input_val - mean_val) * inv_std;
  normalized_data[idx] = normalized_val;

  if (affine) {
    const T gamma_val = gamma_data[c];
    const T beta_val = beta_data[c];
    output_data[idx] = gamma_val * normalized_val + beta_val;
  } else {
    output_data[idx] = normalized_val;
  }
}

template <typename T>
__global__ void compute_affine_gradients_kernel(const T *gradient_data, const T *normalized_data,
                                                T *gamma_grad, T *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T gamma_sum = T(0);
  T beta_sum = T(0);

  for (size_t n = 0; n < batch_size; ++n) {
    const size_t base_idx = n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      size_t idx = base_idx + i;
      gamma_sum += gradient_data[idx] * normalized_data[idx];
      beta_sum += gradient_data[idx];
    }
  }

  atomicAdd(&gamma_grad[c], gamma_sum);
  atomicAdd(&beta_grad[c], beta_sum);
}

template <typename T>
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_channel_mean_kernel<<<num_blocks, threads_per_block>>>(input_data, mean_data, batch_size,
                                                                 channels, spatial_size);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_channel_variance_kernel<<<num_blocks, threads_per_block>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
  cudaDeviceSynchronize();
}

template <typename T>
void normalize_and_scale_optimized(const T *input_data, const T *mean_data, const T *std_data,
                                   const T *gamma_data, const T *beta_data, T *output_data,
                                   T *normalized_data, size_t batch_size, size_t channels,
                                   size_t spatial_size, bool affine) {
  int total_elements = batch_size * channels * spatial_size;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  normalize_and_scale_kernel<<<num_blocks, threads_per_block>>>(
      input_data, mean_data, std_data, gamma_data, beta_data, output_data, normalized_data,
      batch_size, channels, spatial_size, affine);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_affine_gradients_optimized(const T *gradient_data, const T *normalized_data,
                                        T *gamma_grad, T *beta_grad, size_t batch_size,
                                        size_t channels, size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_affine_gradients_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, normalized_data, gamma_grad, beta_grad, batch_size, channels, spatial_size);
  cudaDeviceSynchronize();
}

// Explicit template instantiations
template void compute_channel_mean<float>(const float *input_data, float *mean_data,
                                          size_t batch_size, size_t channels, size_t spatial_size);
template void compute_channel_mean<double>(const double *input_data, double *mean_data,
                                           size_t batch_size, size_t channels, size_t spatial_size);

template void compute_channel_variance<float>(const float *input_data, const float *mean_data,
                                              float *var_data, size_t batch_size, size_t channels,
                                              size_t spatial_size);
template void compute_channel_variance<double>(const double *input_data, const double *mean_data,
                                               double *var_data, size_t batch_size, size_t channels,
                                               size_t spatial_size);

template void normalize_and_scale_optimized<float>(const float *input_data, const float *mean_data,
                                                   const float *std_data, const float *gamma_data,
                                                   const float *beta_data, float *output_data,
                                                   float *normalized_data, size_t batch_size,
                                                   size_t channels, size_t spatial_size,
                                                   bool affine);
template void normalize_and_scale_optimized<double>(
    const double *input_data, const double *mean_data, const double *std_data,
    const double *gamma_data, const double *beta_data, double *output_data, double *normalized_data,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine);

template void compute_affine_gradients_optimized<float>(const float *gradient_data,
                                                        const float *normalized_data,
                                                        float *gamma_grad, float *beta_grad,
                                                        size_t batch_size, size_t channels,
                                                        size_t spatial_size);
template void compute_affine_gradients_optimized<double>(const double *gradient_data,
                                                         const double *normalized_data,
                                                         double *gamma_grad, double *beta_grad,
                                                         size_t batch_size, size_t channels,
                                                         size_t spatial_size);
} // namespace batchnorm
} // namespace cuda
} // namespace tnn