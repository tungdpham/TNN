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

template <typename T>
__global__ void fused_stats_kernel(const T *__restrict__ input, T *__restrict__ mean_out,
                                   T *__restrict__ inv_std_out, T *__restrict__ running_mean,
                                   T *__restrict__ running_var, size_t N, size_t C, size_t S,
                                   T momentum, T epsilon) {

  int c = blockIdx.x;
  if (c >= C)
    return;

  size_t count = N * S;
  T sum = T(0);
  T sq_sum = T(0);

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;
    T val = input[idx];
    sum += val;
    sq_sum += val * val;
  }

  sum = blockReduceSum(sum);
  sq_sum = blockReduceSum(sq_sum);

  if (threadIdx.x == 0) {
    T inv_N = T(1) / T(count);
    T mu = sum * inv_N;
    T var = (sq_sum * inv_N) - (mu * mu);

    mean_out[c] = mu;

    T inv_std = rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;

    running_mean[c] = (T(1) - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (T(1) - momentum) * running_var[c] + momentum * var;
  }
}

template <typename T>
__global__ void fused_apply_kernel(const T *__restrict__ input, const T *__restrict__ mean,
                                   const T *__restrict__ inv_std, const T *__restrict__ gamma,
                                   const T *__restrict__ beta, T *__restrict__ output,
                                   T *__restrict__ normalized_cache, size_t N, size_t C, size_t S,
                                   bool affine) {
  size_t total_elements = N * C * S;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t spatial_size = S;

  if (idx < total_elements) {
    int c = (idx / spatial_size) % C;

    T mu = mean[c];
    T istd = inv_std[c];
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

template <typename T>
__global__ void fused_backward_reduce_kernel(const T *__restrict__ grad_output,
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

    d_gamma[c] = sum_dy_x_norm;
    d_beta[c] = sum_dy;
  }
}

template <typename T>
__global__ void
fused_backward_apply_kernel(const T *__restrict__ grad_output,
                            const T *__restrict__ normalized_input, const T *__restrict__ inv_std,
                            const T *__restrict__ gamma, const T *__restrict__ d_gamma,
                            const T *__restrict__ d_beta, T *__restrict__ grad_input, size_t N,
                            size_t C, size_t S, bool affine) {
  size_t total_elements = N * C * S;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    T g = affine ? gamma[c] : T(1);
    T istd = inv_std[c];
    T sum_dy = d_beta[c];
    T sum_dy_x_norm = d_gamma[c];
    T M = T(N * S);

    T dy = grad_output[idx];
    T x_hat = normalized_input[idx];

    T term1 = (g * istd) / M;
    T term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);

    grad_input[idx] = term1 * term2;
  }
}

template <typename T>
void run_forward_fused(const T *input, T *mean, T *inv_std, T *running_mean, T *running_var,
                       const T *gamma, const T *beta, T *output, T *norm_cache, size_t N, size_t C,
                       size_t S, T momentum, T epsilon, bool affine, cudaStream_t stream) {

  fused_stats_kernel<<<C, BLOCK_SIZE, 0, stream>>>(input, mean, inv_std, running_mean, running_var,
                                                   N, C, S, momentum, epsilon);

  size_t total_elements = N * C * S;
  int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_apply_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, mean, inv_std, gamma, beta,
                                                            output, norm_cache, N, C, S, affine);
}

template <typename T>
void run_backward_fused(const T *grad_output, const T *norm_input, const T *inv_std, const T *gamma,
                        T *d_gamma, T *d_beta, T *grad_input, size_t N, size_t C, size_t S,
                        bool affine, cudaStream_t stream) {

  fused_backward_reduce_kernel<<<C, BLOCK_SIZE, 0, stream>>>(grad_output, norm_input, d_gamma,
                                                             d_beta, N, C, S);

  size_t total_elements = N * C * S;
  int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_backward_apply_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      grad_output, norm_input, inv_std, gamma, d_gamma, d_beta, grad_input, N, C, S, affine);
}

template <typename T>
__global__ void compute_channel_mean_kernel(const T *input_data, T *mean_data, size_t batch_size,
                                            size_t channels, size_t spatial_size) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements_per_channel);
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;
    thread_sum += input_data[global_idx];
  }

  thread_sum = blockReduceSum(thread_sum);

  if (threadIdx.x == 0) {
    mean_data[c] = thread_sum * inv_total;
  }
}

template <typename T>
__global__ void compute_channel_variance_kernel(const T *input_data, const T *mean_data,
                                                T *var_data, size_t batch_size, size_t channels,
                                                size_t spatial_size) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements_per_channel);
  const T mean_val = mean_data[c];
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_sq = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    T diff = input_data[global_idx] - mean_val;
    thread_sum_sq += diff * diff;
  }

  thread_sum_sq = blockReduceSum(thread_sum_sq);

  if (threadIdx.x == 0) {
    var_data[c] = thread_sum_sq * inv_total;
  }
}

template <typename T>
__global__ void compute_affine_gradients_kernel(const T *gradient_data, const T *normalized_data,
                                                T *gamma_grad, T *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size) {

  int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_gamma_sum = T(0);
  T thread_beta_sum = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t idx = n * channel_stride + c_offset + i_local;

    thread_gamma_sum += gradient_data[idx] * normalized_data[idx];
    thread_beta_sum += gradient_data[idx];
  }

  thread_gamma_sum = blockReduceSum(thread_gamma_sum);
  thread_beta_sum = blockReduceSum(thread_beta_sum);

  if (threadIdx.x == 0) {
    atomicAdd(&gamma_grad[c], thread_gamma_sum);
    atomicAdd(&beta_grad[c], thread_beta_sum);
  }
}

template <typename T>
__global__ void normalize_and_scalekernel(const T *input_data, const T *mean_data,
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

template <typename T>
__global__ void compute_mean_variance_fused_kernel(const T *input_data, T *mean_data, T *var_data,
                                                   size_t batch_size, size_t channels,
                                                   size_t spatial_size) {
  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum = T(0);
  T thread_sum_sq = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;
    const T val = input_data[global_idx];
    thread_sum += val;
    thread_sum_sq += val * val;
  }

  thread_sum = blockReduceSum(thread_sum);
  thread_sum_sq = blockReduceSum(thread_sum_sq);

  if (threadIdx.x == 0) {
    const T inv_n = T(1) / static_cast<T>(total_elements_per_channel);
    const T mean_val = thread_sum * inv_n;
    const T mean_sq = thread_sum_sq * inv_n;
    mean_data[c] = mean_val;
    var_data[c] = mean_sq - mean_val * mean_val;
  }
}

template <typename T>
void compute_mean_variance_fused(const T *input_data, T *mean_data, T *var_data, size_t batch_size,
                                 size_t channels, size_t spatial_size, cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_mean_variance_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
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

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_grad_norm = T(0);
  T thread_sum_grad_norm_x_norm = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    thread_sum_grad_norm += grad_normalized_data[global_idx];
    thread_sum_grad_norm_x_norm += grad_normalized_data[global_idx] * normalized_data[global_idx];
  }

  thread_sum_grad_norm = blockReduceSum(thread_sum_grad_norm);
  thread_sum_grad_norm_x_norm = blockReduceSum(thread_sum_grad_norm_x_norm);

  if (threadIdx.x == 0) {
    sum_grad_normalized_data[c] = thread_sum_grad_norm;
    sum_grad_norm_times_norm_data[c] = thread_sum_grad_norm_x_norm;
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

template <typename T>
__global__ void compute_batchnorm_backward_sums_fused_kernel(
    const T *gradient_data, const T *normalized_data, const T *gamma_data,
    T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data, T *gamma_grad, T *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_grad_norm = T(0);
  T thread_sum_grad_norm_x_norm = T(0);
  T thread_gamma_sum = T(0);
  T thread_beta_sum = T(0);

  const T gamma_val = affine ? gamma_data[c] : T(1);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
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

  thread_sum_grad_norm = blockReduceSum(thread_sum_grad_norm);
  thread_sum_grad_norm_x_norm = blockReduceSum(thread_sum_grad_norm_x_norm);

  if (affine) {
    thread_gamma_sum = blockReduceSum(thread_gamma_sum);
    thread_beta_sum = blockReduceSum(thread_beta_sum);
  }

  if (threadIdx.x == 0) {
    sum_grad_normalized_data[c] = thread_sum_grad_norm;
    sum_grad_norm_times_norm_data[c] = thread_sum_grad_norm_x_norm;
    if (affine) {
      atomicAdd(&gamma_grad[c], thread_gamma_sum);
      atomicAdd(&beta_grad[c], thread_beta_sum);
    }
  }
}

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
                                      size_t channels, size_t spatial_size, bool affine,
                                      cudaStream_t stream, T *workspace_sum_grad_normalized,
                                      T *workspace_sum_grad_norm_times_norm) {
  const size_t total_elements = batch_size * spatial_size;

  T *sum_grad_normalized_data;
  T *sum_grad_norm_times_norm_data;
  bool need_cleanup = false;

  if (workspace_sum_grad_normalized && workspace_sum_grad_norm_times_norm) {
    sum_grad_normalized_data = workspace_sum_grad_normalized;
    sum_grad_norm_times_norm_data = workspace_sum_grad_norm_times_norm;
  } else {
    cudaMallocAsync(&sum_grad_normalized_data, channels * sizeof(T), stream);
    cudaMallocAsync(&sum_grad_norm_times_norm_data, channels * sizeof(T), stream);
    need_cleanup = true;
  }

  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_batchnorm_backward_sums_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, normalized_data, gamma_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, gamma_grad, beta_grad, batch_size, channels, spatial_size,
      affine);

  size_t total_size = batch_size * channels * spatial_size;
  num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, normalized_data, std_data, gamma_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements, affine);

  if (need_cleanup) {
    cudaFreeAsync(sum_grad_normalized_data, stream);
    cudaFreeAsync(sum_grad_norm_times_norm_data, stream);
  }
}

template <typename T>
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size, cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_channel_mean_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, batch_size, channels, spatial_size);
}

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size,
                              cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_channel_variance_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
}

template <typename T>
void normalize_and_scale(const T *input_data, const T *mean_data, const T *std_data,
                         const T *gamma_data, const T *beta_data, T *output_data,
                         T *normalized_data, size_t batch_size, size_t channels,
                         size_t spatial_size, bool affine, cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  normalize_and_scalekernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, std_data, gamma_data, beta_data, output_data, normalized_data,
      batch_size, channels, spatial_size, affine);
}

template <typename T>
void compute_affine_gradients_(const T *gradient_data, const T *normalized_data, T *gamma_grad,
                               T *beta_grad, size_t batch_size, size_t channels,
                               size_t spatial_size, cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_affine_gradients_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, normalized_data, gamma_grad, beta_grad, batch_size, channels, spatial_size);
}

template <typename T>
void compute_batch_std(const T *batch_var_data, T *batch_std_data, size_t channels, T epsilon,
                       cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_batch_std_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      batch_var_data, batch_std_data, channels, epsilon);
}

template <typename T>
void update_running_stats(T *running_mean_data, T *running_var_data, const T *batch_mean_data,
                          const T *batch_var_data, size_t channels, T momentum,
                          cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  update_running_stats_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      running_mean_data, running_var_data, batch_mean_data, batch_var_data, channels, momentum);
}

template <typename T>
void compute_inference_output(const T *input_data, const T *running_mean_data,
                              const T *running_var_data, const T *gamma_data, const T *beta_data,
                              T *output_data, size_t batch_size, size_t channels,
                              size_t spatial_size, T epsilon, bool affine, cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_inference_output_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, running_mean_data, running_var_data, gamma_data, beta_data, output_data,
      batch_size, channels, spatial_size, epsilon, affine);
}

template <typename T>
void compute_grad_normalized(const T *gradient_data, const T *gamma_data, T *grad_normalized_data,
                             size_t batch_size, size_t channels, size_t spatial_size, bool affine,
                             cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_grad_normalized_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, gamma_data, grad_normalized_data, batch_size, channels, spatial_size, affine);
}

template <typename T>
void compute_backward_sums(const T *grad_normalized_data, const T *normalized_data,
                           T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data,
                           size_t batch_size, size_t channels, size_t spatial_size,
                           cudaStream_t stream) {

  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_backward_sums_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      grad_normalized_data, normalized_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, batch_size, channels, spatial_size);
}

template <typename T>
void compute_input_gradients_batchnorm(const T *grad_normalized_data, const T *normalized_data,
                                       const T *std_data, const T *sum_grad_normalized_data,
                                       const T *sum_grad_norm_times_norm_data, T *grad_input_data,
                                       size_t batch_size, size_t channels, size_t spatial_size,
                                       size_t total_elements, cudaStream_t stream) {
  size_t total_size = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      grad_normalized_data, normalized_data, std_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements);
}

template void compute_channel_mean<float>(const float *input_data, float *mean_data,
                                          size_t batch_size, size_t channels, size_t spatial_size,
                                          cudaStream_t stream);
template void compute_channel_mean<double>(const double *input_data, double *mean_data,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           cudaStream_t stream);

template void compute_channel_variance<float>(const float *input_data, const float *mean_data,
                                              float *var_data, size_t batch_size, size_t channels,
                                              size_t spatial_size, cudaStream_t stream);
template void compute_channel_variance<double>(const double *input_data, const double *mean_data,
                                               double *var_data, size_t batch_size, size_t channels,
                                               size_t spatial_size, cudaStream_t stream);

template void compute_mean_variance_fused<float>(const float *input_data, float *mean_data,
                                                 float *var_data, size_t batch_size,
                                                 size_t channels, size_t spatial_size,
                                                 cudaStream_t stream);
template void compute_mean_variance_fused<double>(const double *input_data, double *mean_data,
                                                  double *var_data, size_t batch_size,
                                                  size_t channels, size_t spatial_size,
                                                  cudaStream_t stream);

template void normalize_and_scale<float>(const float *input_data, const float *mean_data,
                                         const float *std_data, const float *gamma_data,
                                         const float *beta_data, float *output_data,
                                         float *normalized_data, size_t batch_size, size_t channels,
                                         size_t spatial_size, bool affine, cudaStream_t stream);
template void normalize_and_scale<double>(const double *input_data, const double *mean_data,
                                          const double *std_data, const double *gamma_data,
                                          const double *beta_data, double *output_data,
                                          double *normalized_data, size_t batch_size,
                                          size_t channels, size_t spatial_size, bool affine,
                                          cudaStream_t stream);

template void compute_batch_std<float>(const float *batch_var_data, float *batch_std_data,
                                       size_t channels, float epsilon, cudaStream_t stream);
template void compute_batch_std<double>(const double *batch_var_data, double *batch_std_data,
                                        size_t channels, double epsilon, cudaStream_t stream);

template void update_running_stats<float>(float *running_mean_data, float *running_var_data,
                                          const float *batch_mean_data, const float *batch_var_data,
                                          size_t channels, float momentum, cudaStream_t stream);
template void update_running_stats<double>(double *running_mean_data, double *running_var_data,
                                           const double *batch_mean_data,
                                           const double *batch_var_data, size_t channels,
                                           double momentum, cudaStream_t stream);

template void compute_inference_output<float>(
    const float *input_data, const float *running_mean_data, const float *running_var_data,
    const float *gamma_data, const float *beta_data, float *output_data, size_t batch_size,
    size_t channels, size_t spatial_size, float epsilon, bool affine, cudaStream_t stream);
template void compute_inference_output<double>(
    const double *input_data, const double *running_mean_data, const double *running_var_data,
    const double *gamma_data, const double *beta_data, double *output_data, size_t batch_size,
    size_t channels, size_t spatial_size, double epsilon, bool affine, cudaStream_t stream);

template void compute_affine_gradients_<float>(const float *gradient_data,
                                               const float *normalized_data, float *gamma_grad,
                                               float *beta_grad, size_t batch_size, size_t channels,
                                               size_t spatial_size, cudaStream_t stream);
template void compute_affine_gradients_<double>(const double *gradient_data,
                                                const double *normalized_data, double *gamma_grad,
                                                double *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size,
                                                cudaStream_t stream);

template void compute_grad_normalized<float>(const float *gradient_data, const float *gamma_data,
                                             float *grad_normalized_data, size_t batch_size,
                                             size_t channels, size_t spatial_size, bool affine,
                                             cudaStream_t stream);
template void compute_grad_normalized<double>(const double *gradient_data, const double *gamma_data,
                                              double *grad_normalized_data, size_t batch_size,
                                              size_t channels, size_t spatial_size, bool affine,
                                              cudaStream_t stream);

template void compute_backward_sums<float>(const float *grad_normalized_data,
                                           const float *normalized_data,
                                           float *sum_grad_normalized_data,
                                           float *sum_grad_norm_times_norm_data, size_t batch_size,
                                           size_t channels, size_t spatial_size,
                                           cudaStream_t stream);
template void compute_backward_sums<double>(const double *grad_normalized_data,
                                            const double *normalized_data,
                                            double *sum_grad_normalized_data,
                                            double *sum_grad_norm_times_norm_data,
                                            size_t batch_size, size_t channels, size_t spatial_size,
                                            cudaStream_t stream);

template void compute_input_gradients_batchnorm<float>(
    const float *grad_normalized_data, const float *normalized_data, const float *std_data,
    const float *sum_grad_normalized_data, const float *sum_grad_norm_times_norm_data,
    float *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements, cudaStream_t stream);
template void compute_input_gradients_batchnorm<double>(
    const double *grad_normalized_data, const double *normalized_data, const double *std_data,
    const double *sum_grad_normalized_data, const double *sum_grad_norm_times_norm_data,
    double *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements, cudaStream_t stream);

template void compute_batchnorm_backward_fused<float>(
    const float *gradient_data, const float *normalized_data, const float *std_data,
    const float *gamma_data, float *grad_input_data, float *gamma_grad, float *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine, cudaStream_t stream,
    float *workspace_sum_grad_normalized, float *workspace_sum_grad_norm_times_norm);
template void compute_batchnorm_backward_fused<double>(
    const double *gradient_data, const double *normalized_data, const double *std_data,
    const double *gamma_data, double *grad_input_data, double *gamma_grad, double *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine, cudaStream_t stream,
    double *workspace_sum_grad_normalized, double *workspace_sum_grad_norm_times_norm);

template void run_forward_fused<float>(const float *input, float *mean, float *inv_std,
                                       float *running_mean, float *running_var, const float *gamma,
                                       const float *beta, float *output, float *norm_cache,
                                       size_t N, size_t C, size_t S, float momentum, float epsilon,
                                       bool affine, cudaStream_t stream);
template void run_forward_fused<double>(const double *input, double *mean, double *inv_std,
                                        double *running_mean, double *running_var,
                                        const double *gamma, const double *beta, double *output,
                                        double *norm_cache, size_t N, size_t C, size_t S,
                                        double momentum, double epsilon, bool affine,
                                        cudaStream_t stream);

template void run_backward_fused<float>(const float *grad_output, const float *norm_input,
                                        const float *inv_std, const float *gamma, float *d_gamma,
                                        float *d_beta, float *grad_input, size_t N, size_t C,
                                        size_t S, bool affine, cudaStream_t stream);
template void run_backward_fused<double>(const double *grad_output, const double *norm_input,
                                         const double *inv_std, const double *gamma,
                                         double *d_gamma, double *d_beta, double *grad_input,
                                         size_t N, size_t C, size_t S, bool affine,
                                         cudaStream_t stream);

} // namespace batchnorm
} // namespace cuda
} // namespace tnn