/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/groupnorm_ops.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace groupnorm {

#define BLOCK_SIZE 256
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

template <typename T> __inline__ __device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T> __inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
  if (wid == 0)
    val = warpReduceSum(val);

  return val;
}

// Compute statistics per group: each block handles one group (n, g)
template <typename T>
__global__ void fused_group_stats_kernel(const T *__restrict__ input, T *__restrict__ mean_out,
                                         T *__restrict__ inv_std_out, size_t N, size_t C, size_t S,
                                         size_t num_groups, T epsilon) {

  size_t group_idx = blockIdx.x; // Each block handles one (n, g) pair
  size_t n = group_idx / num_groups;
  size_t g = group_idx % num_groups;

  if (n >= N || g >= num_groups)
    return;

  size_t channels_per_group = C / num_groups;
  size_t group_size = channels_per_group * S;
  size_t channel_stride = C * S;

  T sum = T(0);

  // First pass: compute mean for this group
  for (size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    size_t c_in_group = i / S;
    size_t s = i % S;
    size_t global_c = g * channels_per_group + c_in_group;
    size_t idx = n * channel_stride + global_c * S + s;
    sum += input[idx];
  }

  sum = blockReduceSum(sum);

  // Broadcast mean to all threads
  __shared__ T shared_mean;
  if (threadIdx.x == 0) {
    T inv_group_size = T(1) / T(group_size);
    T mu = sum * inv_group_size;
    shared_mean = mu;
    mean_out[group_idx] = mu;
  }
  __syncthreads();
  T mu = shared_mean;

  // Second pass: compute variance
  T var_sum = T(0);
  for (size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    size_t c_in_group = i / S;
    size_t s = i % S;
    size_t global_c = g * channels_per_group + c_in_group;
    size_t idx = n * channel_stride + global_c * S + s;
    T diff = input[idx] - mu;
    var_sum += diff * diff;
  }

  var_sum = blockReduceSum(var_sum);

  if (threadIdx.x == 0) {
    T inv_group_size = T(1) / T(group_size);
    T var = var_sum * inv_group_size;
    T inv_std = rsqrt(var + epsilon);
    inv_std_out[group_idx] = inv_std;
  }
}

// Apply normalization: each thread handles one element
template <typename T>
__global__ void fused_group_apply_kernel(const T *__restrict__ input, const T *__restrict__ mean,
                                         const T *__restrict__ inv_std, const T *__restrict__ gamma,
                                         const T *__restrict__ beta, T *__restrict__ output,
                                         T *__restrict__ normalized_cache, size_t N, size_t C,
                                         size_t S, size_t num_groups, bool affine) {
  size_t total_elements = N * C * S;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_elements) {
    size_t channels_per_group = C / num_groups;

    size_t n = idx / (C * S);
    size_t c = (idx / S) % C;
    size_t g = c / channels_per_group;
    size_t group_idx = n * num_groups + g;

    T mu = mean[group_idx];
    T istd = inv_std[group_idx];
    T x = input[idx];

    T norm = (x - mu) * istd;

    if (normalized_cache)
      normalized_cache[idx] = norm;

    T res = norm;
    if (affine) {
      res = res * gamma[c] + beta[c];
    }
    output[idx] = res;
  }
}

// Reduce gradients per channel for parameter updates
template <typename T>
__global__ void fused_group_backward_reduce_kernel(const T *__restrict__ grad_output,
                                                   const T *__restrict__ normalized_input,
                                                   T *__restrict__ d_gamma, T *__restrict__ d_beta,
                                                   size_t N, size_t C, size_t S) {

  int c = blockIdx.x;
  if (c >= C)
    return;

  size_t count = N * S;
  T sum_dy = T(0);
  T sum_dy_x_norm = T(0);

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;

    T dy = grad_output[idx];
    T x_hat = normalized_input[idx];

    sum_dy += dy;
    sum_dy_x_norm += dy * x_hat;
  }

  sum_dy = blockReduceSum(sum_dy);
  sum_dy_x_norm = blockReduceSum(sum_dy_x_norm);

  if (threadIdx.x == 0) {
    d_gamma[c] += sum_dy_x_norm;
    d_beta[c] += sum_dy;
  }
}

// Compute input gradients per group
template <typename T>
__global__ void fused_group_backward_apply_kernel(const T *__restrict__ grad_output,
                                                  const T *__restrict__ normalized_input,
                                                  const T *__restrict__ inv_std,
                                                  const T *__restrict__ gamma,
                                                  T *__restrict__ grad_input, size_t N, size_t C,
                                                  size_t S, size_t num_groups, bool affine) {

  size_t group_idx = blockIdx.x; // Each block handles one (n, g) pair
  size_t n = group_idx / num_groups;
  size_t g = group_idx % num_groups;

  if (n >= N || g >= num_groups)
    return;

  size_t channels_per_group = C / num_groups;
  size_t group_size = channels_per_group * S;
  size_t channel_stride = C * S;

  T istd = inv_std[group_idx];

  // Compute sums over the group
  T sum_dy = T(0);
  T sum_dy_x_norm = T(0);

  for (size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    size_t c_in_group = i / S;
    size_t s = i % S;
    size_t global_c = g * channels_per_group + c_in_group;
    size_t idx = n * channel_stride + global_c * S + s;

    T dy = grad_output[idx];
    T x_hat = normalized_input[idx];
    T gamma_val = (affine && gamma) ? gamma[global_c] : T(1);

    sum_dy += dy * gamma_val;
    sum_dy_x_norm += dy * gamma_val * x_hat;
  }

  sum_dy = blockReduceSum(sum_dy);
  sum_dy_x_norm = blockReduceSum(sum_dy_x_norm);

  // Broadcast sums
  __shared__ T shared_sum_dy;
  __shared__ T shared_sum_dy_x_norm;
  if (threadIdx.x == 0) {
    shared_sum_dy = sum_dy;
    shared_sum_dy_x_norm = sum_dy_x_norm;
  }
  __syncthreads();
  sum_dy = shared_sum_dy;
  sum_dy_x_norm = shared_sum_dy_x_norm;

  // Compute gradients for each element in the group
  T inv_group_size = T(1) / T(group_size);

  for (size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    size_t c_in_group = i / S;
    size_t s = i % S;
    size_t global_c = g * channels_per_group + c_in_group;
    size_t idx = n * channel_stride + global_c * S + s;

    T gamma_val = (affine && gamma) ? gamma[global_c] : T(1);
    T dy = grad_output[idx];
    T x_hat = normalized_input[idx];

    T term1 = (gamma_val * istd) * inv_group_size;
    T term2 = T(group_size) * dy * gamma_val - sum_dy - (x_hat * sum_dy_x_norm);

    grad_input[idx] = term1 * term2;
  }
}

template <typename T>
void run_forward_fused(const T *input, T *mean, T *inv_std, const T *gamma, const T *beta,
                       T *output, T *norm_cache, size_t N, size_t C, size_t S, size_t num_groups,
                       T epsilon, bool affine, cudaStream_t stream) {

  size_t total_groups = N * num_groups;
  fused_group_stats_kernel<<<total_groups, BLOCK_SIZE, 0, stream>>>(input, mean, inv_std, N, C, S,
                                                                    num_groups, epsilon);

  size_t total_elements = N * C * S;
  int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_group_apply_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      input, mean, inv_std, gamma, beta, output, norm_cache, N, C, S, num_groups, affine);
}

template <typename T>
void run_backward_fused(const T *grad_output, const T *norm_input, const T *inv_std, const T *gamma,
                        T *d_gamma, T *d_beta, T *grad_input, size_t N, size_t C, size_t S,
                        size_t num_groups, bool affine, cudaStream_t stream) {

  if (affine) {
    fused_group_backward_reduce_kernel<<<C, BLOCK_SIZE, 0, stream>>>(grad_output, norm_input,
                                                                     d_gamma, d_beta, N, C, S);
  }

  size_t total_groups = N * num_groups;
  fused_group_backward_apply_kernel<<<total_groups, BLOCK_SIZE, 0, stream>>>(
      grad_output, norm_input, inv_std, gamma, grad_input, N, C, S, num_groups, affine);
}

template void run_forward_fused<float>(const float *input, float *mean, float *inv_std,
                                       const float *gamma, const float *beta, float *output,
                                       float *norm_cache, size_t N, size_t C, size_t S,
                                       size_t num_groups, float epsilon, bool affine,
                                       cudaStream_t stream);
template void run_forward_fused<double>(const double *input, double *mean, double *inv_std,
                                        const double *gamma, const double *beta, double *output,
                                        double *norm_cache, size_t N, size_t C, size_t S,
                                        size_t num_groups, double epsilon, bool affine,
                                        cudaStream_t stream);

template void run_backward_fused<float>(const float *grad_output, const float *norm_input,
                                        const float *inv_std, const float *gamma, float *d_gamma,
                                        float *d_beta, float *grad_input, size_t N, size_t C,
                                        size_t S, size_t num_groups, bool affine,
                                        cudaStream_t stream);
template void run_backward_fused<double>(const double *grad_output, const double *norm_input,
                                         const double *inv_std, const double *gamma,
                                         double *d_gamma, double *d_beta, double *grad_input,
                                         size_t N, size_t C, size_t S, size_t num_groups,
                                         bool affine, cudaStream_t stream);

} // namespace groupnorm
} // namespace cuda
} // namespace tnn
