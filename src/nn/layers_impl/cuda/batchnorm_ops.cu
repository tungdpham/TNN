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

#define THREADS_PER_BLOCK 256

template <typename T>
__global__ void compute_channel_mean_kernel(const T *input_data, T *mean_data, size_t batch_size,
                                            size_t channels, size_t spatial_size) {

  union SharedData {
    T sdata[THREADS_PER_BLOCK];
  };
  __shared__ SharedData shared_union;
  T *sdata = shared_union.sdata;

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t block_size = blockDim.x;
  const size_t total_elements_per_channel = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements_per_channel);
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += block_size) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;
    thread_sum += input_data[global_idx];
  }

  sdata[threadIdx.x] = thread_sum;
  __syncthreads();

  for (unsigned int s = block_size / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    mean_data[c] = sdata[0] * inv_total;
  }
}

template <typename T>
__global__ void compute_channel_variance_kernel(const T *input_data, const T *mean_data,
                                                T *var_data, size_t batch_size, size_t channels,
                                                size_t spatial_size) {

  union SharedData {
    T sdata[THREADS_PER_BLOCK];
  };
  __shared__ SharedData shared_union;
  T *sdata = shared_union.sdata;

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t block_size = blockDim.x;
  const size_t total_elements_per_channel = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements_per_channel);
  const T mean_val = mean_data[c];
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_sq = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += block_size) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    T diff = input_data[global_idx] - mean_val;
    thread_sum_sq += diff * diff;
  }

  sdata[threadIdx.x] = thread_sum_sq;
  __syncthreads();

  for (unsigned int s = block_size / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    var_data[c] = sdata[0] * inv_total;
  }
}

template <typename T>
__global__ void compute_affine_gradients_kernel(const T *gradient_data, const T *normalized_data,
                                                T *gamma_grad, T *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size) {

  union SharedData {
    T sdata_gamma_beta[2 * THREADS_PER_BLOCK];
  };
  __shared__ SharedData shared_union;
  T *sdata_gamma_beta = shared_union.sdata_gamma_beta;

  int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t block_size = blockDim.x;
  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T *sdata_gamma = sdata_gamma_beta;
  T *sdata_beta = sdata_gamma_beta + block_size;

  T thread_gamma_sum = T(0);
  T thread_beta_sum = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += block_size) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t idx = n * channel_stride + c_offset + i_local;

    thread_gamma_sum += gradient_data[idx] * normalized_data[idx];
    thread_beta_sum += gradient_data[idx];
  }

  sdata_gamma[threadIdx.x] = thread_gamma_sum;
  sdata_beta[threadIdx.x] = thread_beta_sum;
  __syncthreads();

  for (unsigned int s = block_size / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      sdata_gamma[threadIdx.x] += sdata_gamma[threadIdx.x + s];
      sdata_beta[threadIdx.x] += sdata_beta[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&gamma_grad[c], sdata_gamma[0]);
    atomicAdd(&beta_grad[c], sdata_beta[0]);
  }
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
__global__ void compute_batch_std_kernel(const T *batch_var_data, T *batch_std_data,
                                         size_t channels, T epsilon) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    batch_std_data[c] = sqrt(batch_var_data[c] + epsilon);
  }
}

// Fused mean and variance computation using Welford's online algorithm
template <typename T>
__global__ void compute_mean_variance_fused_kernel(const T *input_data, T *mean_data, T *var_data,
                                                   size_t batch_size, size_t channels,
                                                   size_t spatial_size) {
  union SharedData {
    T sdata_mean[THREADS_PER_BLOCK];
    T sdata_var[THREADS_PER_BLOCK];
  };
  __shared__ T shared_mean[THREADS_PER_BLOCK];
  __shared__ T shared_m2[THREADS_PER_BLOCK];

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t block_size = blockDim.x;
  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  // Welford's algorithm for parallel mean and variance
  T thread_mean = T(0);
  T thread_m2 = T(0);
  size_t thread_count = 0;

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += block_size) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    thread_count++;
    const T val = input_data[global_idx];
    const T delta = val - thread_mean;
    thread_mean += delta / static_cast<T>(thread_count);
    const T delta2 = val - thread_mean;
    thread_m2 += delta * delta2;
  }

  shared_mean[threadIdx.x] = thread_mean;
  shared_m2[threadIdx.x] = thread_m2;
  __syncthreads();

  // Combine results from all threads using Welford's combining formula
  for (unsigned int s = block_size / 2; s > 0; s /= 2) {
    if (threadIdx.x < s && threadIdx.x + s < block_size) {
      const size_t count_a = (threadIdx.x + 1) * (total_elements_per_channel / block_size);
      const size_t count_b = (threadIdx.x + s + 1) * (total_elements_per_channel / block_size);
      const T mean_a = shared_mean[threadIdx.x];
      const T mean_b = shared_mean[threadIdx.x + s];
      const T m2_a = shared_m2[threadIdx.x];
      const T m2_b = shared_m2[threadIdx.x + s];

      const size_t count_combined = count_a + count_b;
      const T delta = mean_b - mean_a;
      const T mean_combined =
          (mean_a * static_cast<T>(count_a) + mean_b * static_cast<T>(count_b)) /
          static_cast<T>(count_combined);
      const T m2_combined = m2_a + m2_b +
                            delta * delta * static_cast<T>(count_a) * static_cast<T>(count_b) /
                                static_cast<T>(count_combined);

      shared_mean[threadIdx.x] = mean_combined;
      shared_m2[threadIdx.x] = m2_combined;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    mean_data[c] = shared_mean[0];
    var_data[c] = shared_m2[0] / static_cast<T>(total_elements_per_channel);
  }
}

template <typename T>
void compute_mean_variance_fused(const T *input_data, T *mean_data, T *var_data, size_t batch_size,
                                 size_t channels, size_t spatial_size) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_mean_variance_fused_kernel<<<num_blocks, threads_per_block>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
}

template <typename T>
__global__ void update_running_stats_kernel(T *running_mean_data, T *running_var_data,
                                            const T *batch_mean_data, const T *batch_var_data,
                                            size_t channels, T momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    running_mean_data[c] = (T(1) - momentum) * running_mean_data[c] + momentum * batch_mean_data[c];
    running_var_data[c] = (T(1) - momentum) * running_var_data[c] + momentum * batch_var_data[c];
  }
}

template <typename T>
__global__ void compute_inference_output_kernel(const T *input_data, const T *running_mean_data,
                                                const T *running_var_data, const T *gamma_data,
                                                const T *beta_data, T *output_data,
                                                size_t batch_size, size_t channels,
                                                size_t spatial_size, T epsilon, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  T mean_val = running_mean_data[c];
  T var_val = running_var_data[c];
  T std_val = sqrt(var_val + epsilon);
  const T inv_std = T(1) / std_val;

  T input_val = input_data[idx];
  T normalized_val = (input_val - mean_val) * inv_std;

  if (affine) {
    const T gamma_val = gamma_data[c];
    const T beta_val = beta_data[c];
    output_data[idx] = gamma_val * normalized_val + beta_val;
  } else {
    output_data[idx] = normalized_val;
  }
}

template <typename T>
__global__ void compute_grad_normalized_kernel(const T *gradient_data, const T *gamma_data,
                                               T *grad_normalized_data, size_t batch_size,
                                               size_t channels, size_t spatial_size, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  if (affine) {
    int c = (idx / spatial_size) % channels;
    const T gamma_val = gamma_data[c];
    grad_normalized_data[idx] = gradient_data[idx] * gamma_val;
  } else {
    grad_normalized_data[idx] = gradient_data[idx];
  }
}

template <typename T>
__global__ void compute_backward_sums_kernel(const T *grad_normalized_data,
                                             const T *normalized_data, T *sum_grad_normalized_data,
                                             T *sum_grad_norm_times_norm_data, size_t batch_size,
                                             size_t channels, size_t spatial_size) {

  union SharedData {
    T sdata_sums[2 * THREADS_PER_BLOCK];
  };
  __shared__ SharedData shared_union;
  T *sdata_sums = shared_union.sdata_sums;

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t block_size = blockDim.x;
  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T *sdata_grad_norm = sdata_sums;
  T *sdata_grad_norm_x_norm = sdata_sums + block_size;

  T thread_sum_grad_norm = T(0);
  T thread_sum_grad_norm_x_norm = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += block_size) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    thread_sum_grad_norm += grad_normalized_data[global_idx];
    thread_sum_grad_norm_x_norm += grad_normalized_data[global_idx] * normalized_data[global_idx];
  }

  sdata_grad_norm[threadIdx.x] = thread_sum_grad_norm;
  sdata_grad_norm_x_norm[threadIdx.x] = thread_sum_grad_norm_x_norm;
  __syncthreads();

  for (unsigned int s = block_size / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      sdata_grad_norm[threadIdx.x] += sdata_grad_norm[threadIdx.x + s];
      sdata_grad_norm_x_norm[threadIdx.x] += sdata_grad_norm_x_norm[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    sum_grad_normalized_data[c] = sdata_grad_norm[0];
    sum_grad_norm_times_norm_data[c] = sdata_grad_norm_x_norm[0];
  }
}

template <typename T>
__global__ void compute_input_gradients_batchnorm_kernel(
    const T *grad_normalized_data, const T *normalized_data, const T *std_data,
    const T *sum_grad_normalized_data, const T *sum_grad_norm_times_norm_data, T *grad_input_data,
    size_t batch_size, size_t channels, size_t spatial_size, size_t total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * channels * spatial_size;

  if (idx >= total_size)
    return;

  int c = (idx / spatial_size) % channels;

  const T std_val_c = std_data[c];
  const T inv_std = T(1) / std_val_c;
  const T sum_grad_norm = sum_grad_normalized_data[c];
  const T sum_grad_norm_x_norm = sum_grad_norm_times_norm_data[c];
  const T inv_total = T(1) / static_cast<T>(total_elements);

  grad_input_data[idx] = inv_std * inv_total *
                         (static_cast<T>(total_elements) * grad_normalized_data[idx] -
                          sum_grad_norm - normalized_data[idx] * sum_grad_norm_x_norm);
}

// Fused backward kernel: combines affine gradients + backward sums computation
template <typename T>
__global__ void compute_batchnorm_backward_sums_fused_kernel(
    const T *gradient_data, const T *normalized_data, const T *gamma_data,
    T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data, T *gamma_grad, T *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine) {

  union SharedData {
    T sdata[4 * THREADS_PER_BLOCK]; // sum_grad, sum_grad*norm, gamma_grad, beta_grad
  };
  __shared__ SharedData shared_union;
  T *sdata = shared_union.sdata;

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t block_size = blockDim.x;
  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T *sdata_grad_norm = sdata;
  T *sdata_grad_norm_x_norm = sdata + block_size;
  T *sdata_gamma = sdata + 2 * block_size;
  T *sdata_beta = sdata + 3 * block_size;

  T thread_sum_grad_norm = T(0);
  T thread_sum_grad_norm_x_norm = T(0);
  T thread_gamma_sum = T(0);
  T thread_beta_sum = T(0);

  const T gamma_val = affine ? gamma_data[c] : T(1);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += block_size) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    const T grad_val = gradient_data[global_idx];
    const T norm_val = normalized_data[global_idx];
    const T grad_norm_val = grad_val * gamma_val;

    thread_sum_grad_norm += grad_norm_val;
    thread_sum_grad_norm_x_norm += grad_norm_val * norm_val;

    if (affine) {
      thread_gamma_sum += grad_val * norm_val;
      thread_beta_sum += grad_val;
    }
  }

  sdata_grad_norm[threadIdx.x] = thread_sum_grad_norm;
  sdata_grad_norm_x_norm[threadIdx.x] = thread_sum_grad_norm_x_norm;
  sdata_gamma[threadIdx.x] = thread_gamma_sum;
  sdata_beta[threadIdx.x] = thread_beta_sum;
  __syncthreads();

  // Reduction
  for (unsigned int s = block_size / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      sdata_grad_norm[threadIdx.x] += sdata_grad_norm[threadIdx.x + s];
      sdata_grad_norm_x_norm[threadIdx.x] += sdata_grad_norm_x_norm[threadIdx.x + s];
      if (affine) {
        sdata_gamma[threadIdx.x] += sdata_gamma[threadIdx.x + s];
        sdata_beta[threadIdx.x] += sdata_beta[threadIdx.x + s];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    sum_grad_normalized_data[c] = sdata_grad_norm[0];
    sum_grad_norm_times_norm_data[c] = sdata_grad_norm_x_norm[0];
    if (affine) {
      atomicAdd(&gamma_grad[c], sdata_gamma[0]);
      atomicAdd(&beta_grad[c], sdata_beta[0]);
    }
  }
}

// Fused input gradients kernel
template <typename T>
__global__ void compute_input_gradients_batchnorm_fused_kernel(
    const T *gradient_data, const T *normalized_data, const T *std_data, const T *gamma_data,
    const T *sum_grad_normalized_data, const T *sum_grad_norm_times_norm_data, T *grad_input_data,
    size_t batch_size, size_t channels, size_t spatial_size, size_t total_elements, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * channels * spatial_size;

  if (idx >= total_size)
    return;

  int c = (idx / spatial_size) % channels;

  const T std_val_c = std_data[c];
  const T inv_std = T(1) / std_val_c;
  const T sum_grad_norm = sum_grad_normalized_data[c];
  const T sum_grad_norm_x_norm = sum_grad_norm_times_norm_data[c];
  const T inv_total = T(1) / static_cast<T>(total_elements);
  const T gamma_val = affine ? gamma_data[c] : T(1);

  const T grad_norm_val = gradient_data[idx] * gamma_val;

  grad_input_data[idx] = inv_std * inv_total *
                         (static_cast<T>(total_elements) * grad_norm_val - sum_grad_norm -
                          normalized_data[idx] * sum_grad_norm_x_norm);
}

template <typename T>
void compute_batchnorm_backward_fused(const T *gradient_data, const T *normalized_data,
                                      const T *std_data, const T *gamma_data, T *grad_input_data,
                                      T *gamma_grad, T *beta_grad, size_t batch_size,
                                      size_t channels, size_t spatial_size, bool affine) {
  const size_t total_elements = batch_size * spatial_size;

  // Allocate temporary buffers for sums
  T *sum_grad_normalized_data;
  T *sum_grad_norm_times_norm_data;
  cudaMalloc(&sum_grad_normalized_data, channels * sizeof(T));
  cudaMalloc(&sum_grad_norm_times_norm_data, channels * sizeof(T));

  // First pass: compute sums and affine gradients
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;
  size_t shared_mem_size = 4 * threads_per_block * sizeof(T);

  compute_batchnorm_backward_sums_fused_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      gradient_data, normalized_data, gamma_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, gamma_grad, beta_grad, batch_size, channels, spatial_size,
      affine);

  // Second pass: compute input gradients
  size_t total_size = batch_size * channels * spatial_size;
  num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_fused_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, normalized_data, std_data, gamma_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements, affine);

  cudaFree(sum_grad_normalized_data);
  cudaFree(sum_grad_norm_times_norm_data);
}

template <typename T>
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  size_t shared_mem_size = threads_per_block * sizeof(T);

  compute_channel_mean_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      input_data, mean_data, batch_size, channels, spatial_size);
}

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  size_t shared_mem_size = threads_per_block * sizeof(T);

  compute_channel_variance_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
}

template <typename T>
void normalize_and_scale_optimized(const T *input_data, const T *mean_data, const T *std_data,
                                   const T *gamma_data, const T *beta_data, T *output_data,
                                   T *normalized_data, size_t batch_size, size_t channels,
                                   size_t spatial_size, bool affine) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  normalize_and_scale_kernel<<<num_blocks, threads_per_block>>>(
      input_data, mean_data, std_data, gamma_data, beta_data, output_data, normalized_data,
      batch_size, channels, spatial_size, affine);
}

template <typename T>
void compute_affine_gradients_optimized(const T *gradient_data, const T *normalized_data,
                                        T *gamma_grad, T *beta_grad, size_t batch_size,
                                        size_t channels, size_t spatial_size) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  size_t shared_mem_size = 2 * threads_per_block * sizeof(T);

  compute_affine_gradients_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      gradient_data, normalized_data, gamma_grad, beta_grad, batch_size, channels, spatial_size);
}

template <typename T>
void compute_batch_std(const T *batch_var_data, T *batch_std_data, size_t channels, T epsilon) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_batch_std_kernel<<<num_blocks, threads_per_block>>>(batch_var_data, batch_std_data,
                                                              channels, epsilon);
}

template <typename T>
void update_running_stats(T *running_mean_data, T *running_var_data, const T *batch_mean_data,
                          const T *batch_var_data, size_t channels, T momentum) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  update_running_stats_kernel<<<num_blocks, threads_per_block>>>(
      running_mean_data, running_var_data, batch_mean_data, batch_var_data, channels, momentum);
}

template <typename T>
void compute_inference_output(const T *input_data, const T *running_mean_data,
                              const T *running_var_data, const T *gamma_data, const T *beta_data,
                              T *output_data, size_t batch_size, size_t channels,
                              size_t spatial_size, T epsilon, bool affine) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_inference_output_kernel<<<num_blocks, threads_per_block>>>(
      input_data, running_mean_data, running_var_data, gamma_data, beta_data, output_data,
      batch_size, channels, spatial_size, epsilon, affine);
}

template <typename T>
void compute_grad_normalized(const T *gradient_data, const T *gamma_data, T *grad_normalized_data,
                             size_t batch_size, size_t channels, size_t spatial_size, bool affine) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_grad_normalized_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, gamma_data, grad_normalized_data, batch_size, channels, spatial_size, affine);
}

template <typename T>
void compute_backward_sums(const T *grad_normalized_data, const T *normalized_data,
                           T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data,
                           size_t batch_size, size_t channels, size_t spatial_size) {

  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  size_t shared_mem_size = 2 * threads_per_block * sizeof(T);

  compute_backward_sums_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      grad_normalized_data, normalized_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, batch_size, channels, spatial_size);
}

template <typename T>
void compute_input_gradients_batchnorm(const T *grad_normalized_data, const T *normalized_data,
                                       const T *std_data, const T *sum_grad_normalized_data,
                                       const T *sum_grad_norm_times_norm_data, T *grad_input_data,
                                       size_t batch_size, size_t channels, size_t spatial_size,
                                       size_t total_elements) {
  size_t total_size = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_kernel<<<num_blocks, threads_per_block>>>(
      grad_normalized_data, normalized_data, std_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements);
}

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

template void compute_mean_variance_fused<float>(const float *input_data, float *mean_data,
                                                 float *var_data, size_t batch_size,
                                                 size_t channels, size_t spatial_size);
template void compute_mean_variance_fused<double>(const double *input_data, double *mean_data,
                                                  double *var_data, size_t batch_size,
                                                  size_t channels, size_t spatial_size);

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

template void compute_batch_std<float>(const float *batch_var_data, float *batch_std_data,
                                       size_t channels, float epsilon);
template void compute_batch_std<double>(const double *batch_var_data, double *batch_std_data,
                                        size_t channels, double epsilon);

template void update_running_stats<float>(float *running_mean_data, float *running_var_data,
                                          const float *batch_mean_data, const float *batch_var_data,
                                          size_t channels, float momentum);
template void update_running_stats<double>(double *running_mean_data, double *running_var_data,
                                           const double *batch_mean_data,
                                           const double *batch_var_data, size_t channels,
                                           double momentum);

template void
compute_inference_output<float>(const float *input_data, const float *running_mean_data,
                                const float *running_var_data, const float *gamma_data,
                                const float *beta_data, float *output_data, size_t batch_size,
                                size_t channels, size_t spatial_size, float epsilon, bool affine);
template void
compute_inference_output<double>(const double *input_data, const double *running_mean_data,
                                 const double *running_var_data, const double *gamma_data,
                                 const double *beta_data, double *output_data, size_t batch_size,
                                 size_t channels, size_t spatial_size, double epsilon, bool affine);

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

template void compute_grad_normalized<float>(const float *gradient_data, const float *gamma_data,
                                             float *grad_normalized_data, size_t batch_size,
                                             size_t channels, size_t spatial_size, bool affine);
template void compute_grad_normalized<double>(const double *gradient_data, const double *gamma_data,
                                              double *grad_normalized_data, size_t batch_size,
                                              size_t channels, size_t spatial_size, bool affine);

template void compute_backward_sums<float>(const float *grad_normalized_data,
                                           const float *normalized_data,
                                           float *sum_grad_normalized_data,
                                           float *sum_grad_norm_times_norm_data, size_t batch_size,
                                           size_t channels, size_t spatial_size);
template void compute_backward_sums<double>(const double *grad_normalized_data,
                                            const double *normalized_data,
                                            double *sum_grad_normalized_data,
                                            double *sum_grad_norm_times_norm_data,
                                            size_t batch_size, size_t channels,
                                            size_t spatial_size);

template void compute_input_gradients_batchnorm<float>(
    const float *grad_normalized_data, const float *normalized_data, const float *std_data,
    const float *sum_grad_normalized_data, const float *sum_grad_norm_times_norm_data,
    float *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements);
template void compute_input_gradients_batchnorm<double>(
    const double *grad_normalized_data, const double *normalized_data, const double *std_data,
    const double *sum_grad_normalized_data, const double *sum_grad_norm_times_norm_data,
    double *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements);

template void compute_batchnorm_backward_fused<float>(
    const float *gradient_data, const float *normalized_data, const float *std_data,
    const float *gamma_data, float *grad_input_data, float *gamma_grad, float *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine);
template void compute_batchnorm_backward_fused<double>(
    const double *gradient_data, const double *normalized_data, const double *std_data,
    const double *gamma_data, double *grad_input_data, double *gamma_grad, double *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine);

} // namespace batchnorm
} // namespace cuda
} // namespace tnn