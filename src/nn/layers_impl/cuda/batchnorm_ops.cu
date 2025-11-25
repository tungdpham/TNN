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

  size_t stride = C * S;
  size_t offset = c * S;

  // First pass: compute mean
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;
    sum += input[idx];
  }

  sum = blockReduceSum(sum);

  // Broadcast mean to all threads
  __shared__ T shared_mean;
  if (threadIdx.x == 0) {
    T inv_N = T(1) / T(count);
    T mu = sum * inv_N;
    shared_mean = mu;
    mean_out[c] = mu;
  }
  __syncthreads();
  T mu = shared_mean;

  // Second pass: compute variance using stable formula
  T var_sum = T(0);
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;
    T diff = input[idx] - mu;
    var_sum += diff * diff;
  }

  var_sum = blockReduceSum(var_sum);

  if (threadIdx.x == 0) {
    T inv_N = T(1) / T(count);
    T var = var_sum * inv_N;

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

    d_gamma[c] += sum_dy_x_norm;
    d_beta[c] += sum_dy;
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

    T g = (affine && gamma) ? gamma[c] : T(1);
    T istd = inv_std[c];
    T sum_dy = affine ? d_beta[c] : T(0);
    T sum_dy_x_norm = affine ? d_gamma[c] : T(0);
    T M = T(N * S);

    T dy = grad_output[idx];
    T x_hat = normalized_input[idx];

    T term1 = (g * istd) / M;
    T term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);

    grad_input[idx] = term1 * term2;
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

  if (affine) {
    fused_backward_reduce_kernel<<<C, BLOCK_SIZE, 0, stream>>>(grad_output, norm_input, d_gamma,
                                                               d_beta, N, C, S);
  }
  size_t total_elements = N * C * S;
  int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_backward_apply_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      grad_output, norm_input, inv_std, gamma, d_gamma, d_beta, grad_input, N, C, S, affine);
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

template void compute_inference_output<float>(
    const float *input_data, const float *running_mean_data, const float *running_var_data,
    const float *gamma_data, const float *beta_data, float *output_data, size_t batch_size,
    size_t channels, size_t spatial_size, float epsilon, bool affine, cudaStream_t stream);
template void compute_inference_output<double>(
    const double *input_data, const double *running_mean_data, const double *running_var_data,
    const double *gamma_data, const double *beta_data, double *output_data, size_t batch_size,
    size_t channels, size_t spatial_size, double epsilon, bool affine, cudaStream_t stream);

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