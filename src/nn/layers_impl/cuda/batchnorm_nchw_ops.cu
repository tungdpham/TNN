/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/batchnorm_nchw_ops.hpp"

#include "type/type.hpp"
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace batchnorm_nchw {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T> struct VectorType;
template <> struct VectorType<fp16> {
  using type = half2;
  static constexpr int size = 2;
};
template <> struct VectorType<float> {
  using type = float4;
  static constexpr int size = 4;
};
template <> struct VectorType<double> {
  using type = double2;
  static constexpr int size = 2;
};

template <typename T> struct WelfordData {
  T mean;
  T m2;
  T count;

  __device__ WelfordData() : mean(0), m2(0), count(0) {}
  __device__ WelfordData(T m, T v, T c) : mean(m), m2(v), count(c) {}
};

template <typename T> __device__ WelfordData<T> welford_merge(WelfordData<T> a, WelfordData<T> b) {
  if (b.count == T(0))
    return a;
  if (a.count == T(0))
    return b;

  T new_count = a.count + b.count;
  T delta = b.mean - a.mean;
  T new_mean = a.mean + (delta * b.count) / new_count;
  T new_m2 = a.m2 + b.m2 + (delta * delta * a.count * b.count) / new_count;

  return WelfordData<T>(new_mean, new_m2, new_count);
}

template <typename T> __device__ WelfordData<T> warpReduceWelford(WelfordData<T> val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other_mean = __shfl_down_sync(0xffffffff, val.mean, offset);
    T other_m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
    T other_count = __shfl_down_sync(0xffffffff, val.count, offset);
    val = welford_merge(val, WelfordData<T>(other_mean, other_m2, other_count));
  }
  return val;
}

template <typename T> __device__ WelfordData<T> blockReduceWelford(WelfordData<T> val) {
  static __shared__ T shared_mean[32];
  static __shared__ T shared_m2[32];
  static __shared__ T shared_count[32];

  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceWelford(val);

  if (lane == 0) {
    shared_mean[wid] = val.mean;
    shared_m2[wid] = val.m2;
    shared_count[wid] = val.count;
  }
  __syncthreads();

  WelfordData<T> block_val;

  if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
    block_val.mean = shared_mean[threadIdx.x];
    block_val.m2 = shared_m2[threadIdx.x];
    block_val.count = shared_count[threadIdx.x];
  }

  if (wid == 0) {
    block_val = warpReduceWelford(block_val);
  }

  return block_val;
}

template <typename T> __inline__ __device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T> __inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
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
__global__ void fused_stats_kernel(const T *__restrict__ input, float *__restrict__ mean_out,
                                   float *__restrict__ inv_std_out,
                                   float *__restrict__ running_mean,
                                   float *__restrict__ running_var, size_t N, size_t C, size_t S,
                                   float momentum, float epsilon) {
  int c = blockIdx.x;
  if (c >= C)
    return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;

  WelfordData<float> thread_data;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * channel_stride + channel_offset + s;

    float val = static_cast<float>(input[idx]);

    thread_data.count += 1.0f;
    float delta = val - thread_data.mean;
    thread_data.mean += delta / thread_data.count;
    float delta2 = val - thread_data.mean;
    thread_data.m2 += delta * delta2;
  }

  WelfordData<float> result = blockReduceWelford(thread_data);

  if (threadIdx.x == 0) {
    float mu = result.mean;

    float var = result.m2 / result.count;

    mean_out[c] = mu;

    float inv_std = rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;

    float unbiased_var = (result.count > 1.0f) ? (result.m2 / (result.count - 1.0f)) : 0.0f;

    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  }
}

template <typename T>
__global__ void fused_stats_kernel_vec(const T *__restrict__ input, float *__restrict__ mean_out,
                                       float *__restrict__ inv_std_out,
                                       float *__restrict__ running_mean,
                                       float *__restrict__ running_var, size_t N, size_t C,
                                       size_t S, float momentum, float epsilon) {
  using VecT = typename VectorType<T>::type;
  constexpr int vec_size = VectorType<T>::size;

  int c = blockIdx.x;
  if (c >= C)
    return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  WelfordData<float> thread_data;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * channel_stride + channel_offset + s;

    VecT val_vec = *reinterpret_cast<const VecT *>(&input[idx]);
    const T *val_arr = reinterpret_cast<const T *>(&val_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float val = static_cast<float>(val_arr[k]);
      thread_data.count += 1.0f;
      float delta = val - thread_data.mean;
      thread_data.mean += delta / thread_data.count;
      float delta2 = val - thread_data.mean;
      thread_data.m2 += delta * delta2;
    }
  }

  WelfordData<float> result = blockReduceWelford(thread_data);

  if (threadIdx.x == 0) {
    float mu = result.mean;
    float var = result.m2 / result.count;
    mean_out[c] = mu;
    float inv_std = rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;
    float unbiased_var = (result.count > 1.0f) ? (result.m2 / (result.count - 1.0f)) : 0.0f;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  }
}

template <typename T>
__global__ void fused_apply_kernel(const T *__restrict__ input, const float *__restrict__ mean,
                                   const float *__restrict__ inv_std,
                                   const float *__restrict__ gamma, const float *__restrict__ beta,
                                   T *__restrict__ output, float *__restrict__ normalized_cache,
                                   size_t N, size_t C, size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    float mu = mean[c];
    float istd = inv_std[c];
    float x = static_cast<float>(input[idx]);

    float norm = (x - mu) * istd;

    if (normalized_cache)
      normalized_cache[idx] = norm;

    float res = norm;
    if (affine) {
      res = res * gamma[c] + beta[c];
    }
    output[idx] = static_cast<T>(res);
  }
}

template <typename T>
__global__ void fused_apply_kernel_vec(const T *__restrict__ input, const float *__restrict__ mean,
                                       const float *__restrict__ inv_std,
                                       const float *__restrict__ gamma,
                                       const float *__restrict__ beta, T *__restrict__ output,
                                       float *__restrict__ normalized_cache, size_t N, size_t C,
                                       size_t S, bool affine) {
  using VecT = typename VectorType<T>::type;
  constexpr int vec_size = VectorType<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    float mu = mean[c];
    float istd = inv_std[c];
    float g = (affine && gamma) ? gamma[c] : 1.0f;
    float b = (affine && beta) ? beta[c] : 0.0f;

    VecT x_vec = reinterpret_cast<const VecT *>(input)[idx];
    const T *x_arr = reinterpret_cast<const T *>(&x_vec);

    VecT out_vec;
    T *out_arr = reinterpret_cast<T *>(&out_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float x = static_cast<float>(x_arr[k]);
      float norm = (x - mu) * istd;
      if (normalized_cache)
        normalized_cache[scalar_idx + k] = norm;

      float res = norm;
      if (affine) {
        res = res * g + b;
      }
      out_arr[k] = static_cast<T>(res);
    }

    reinterpret_cast<VecT *>(output)[idx] = out_vec;
  }
}

template <typename T>
__global__ void fused_backward_reduce_kernel(const T *__restrict__ grad_output,
                                             const float *__restrict__ normalized_input,
                                             float *__restrict__ d_gamma,
                                             float *__restrict__ d_beta, size_t N, size_t C,
                                             size_t S) {
  int c = blockIdx.x;
  if (c >= C)
    return;

  size_t count = N * S;
  float sum_dy = 0.0f;
  float sum_dy_x_norm = 0.0f;

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;

    float dy = static_cast<float>(grad_output[idx]);
    float x_hat = normalized_input[idx];

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
__global__ void fused_backward_reduce_kernel_vec(const T *__restrict__ grad_output,
                                                 const float *__restrict__ normalized_input,
                                                 float *__restrict__ d_gamma,
                                                 float *__restrict__ d_beta, size_t N, size_t C,
                                                 size_t S) {
  using VecT = typename VectorType<T>::type;
  constexpr int vec_size = VectorType<T>::size;

  int c = blockIdx.x;
  if (c >= C)
    return;

  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  float sum_dy = 0.0f;
  float sum_dy_x_norm = 0.0f;

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * stride + offset + s;

    VecT dy_vec = *reinterpret_cast<const VecT *>(&grad_output[idx]);
    const T *dy_arr = reinterpret_cast<const T *>(&dy_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float dy = static_cast<float>(dy_arr[k]);
      float x_hat = normalized_input[idx + k];
      sum_dy += dy;
      sum_dy_x_norm += dy * x_hat;
    }
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
                            const float *__restrict__ normalized_input,
                            const float *__restrict__ inv_std, const float *__restrict__ gamma,
                            const float *__restrict__ d_gamma, const float *__restrict__ d_beta,
                            T *__restrict__ grad_input, size_t N, size_t C, size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    float g = (affine && gamma) ? gamma[c] : 1.0f;
    float istd = inv_std[c];

    float sum_dy = d_beta[c];
    float sum_dy_x_norm = d_gamma[c];
    float M = static_cast<float>(N * S);

    float dy = static_cast<float>(grad_output[idx]);
    float x_hat = normalized_input[idx];

    float term1 = (g * istd) / M;
    float term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);

    grad_input[idx] = static_cast<T>(term1 * term2);
  }
}

template <typename T>
__global__ void fused_backward_apply_kernel_vec(
    const T *__restrict__ grad_output, const float *__restrict__ normalized_input,
    const float *__restrict__ inv_std, const float *__restrict__ gamma,
    const float *__restrict__ d_gamma, const float *__restrict__ d_beta, T *__restrict__ grad_input,
    size_t N, size_t C, size_t S, bool affine) {
  using VecT = typename VectorType<T>::type;
  constexpr int vec_size = VectorType<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    float g = (affine && gamma) ? gamma[c] : 1.0f;
    float istd = inv_std[c];
    float sum_dy = d_beta[c];
    float sum_dy_x_norm = d_gamma[c];
    float M = static_cast<float>(N * S);

    float term1 = (g * istd) / M;

    VecT dy_vec = reinterpret_cast<const VecT *>(grad_output)[idx];
    const T *dy_arr = reinterpret_cast<const T *>(&dy_vec);

    VecT dx_vec;
    T *dx_arr = reinterpret_cast<T *>(&dx_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float dy = static_cast<float>(dy_arr[k]);
      float x_hat = normalized_input[scalar_idx + k];
      float term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);
      dx_arr[k] = static_cast<T>(term1 * term2);
    }

    reinterpret_cast<VecT *>(grad_input)[idx] = dx_vec;
  }
}

template <typename T>
__global__ void compute_inference_output_kernel(const T *input_data, const float *running_mean_data,
                                                const float *running_var_data,
                                                const float *gamma_data, const float *beta_data,
                                                T *output_data, size_t batch_size, size_t channels,
                                                size_t spatial_size, float epsilon, bool affine) {

  extern __shared__ char shared_mem[];
  float *s_mean = reinterpret_cast<float *>(shared_mem);
  float *s_inv_std = s_mean + channels;
  float *s_gamma = s_inv_std + channels;
  float *s_beta = s_gamma + channels;

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    s_mean[c] = running_mean_data[c];
    float var_val = running_var_data[c];
    s_inv_std[c] = rsqrt(var_val + epsilon);
    if (affine) {
      s_gamma[c] = gamma_data[c];
      s_beta[c] = beta_data[c];
    }
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  float input_val = static_cast<float>(input_data[idx]);
  float normalized_val = (input_val - s_mean[c]) * s_inv_std[c];

  if (affine) {
    output_data[idx] = static_cast<T>(s_gamma[c] * normalized_val + s_beta[c]);
  } else {
    output_data[idx] = static_cast<T>(normalized_val);
  }
}

template <typename T>
void run_forward_fused(const T *input, float *mean, float *inv_std, float *running_mean,
                       float *running_var, const float *gamma, const float *beta, T *output,
                       float *norm_cache, size_t N, size_t C, size_t S, float momentum,
                       float epsilon, bool affine, cudaStream_t stream) {

  constexpr int vec_size = VectorType<T>::size;
  if (S % vec_size == 0) {
    fused_stats_kernel_vec<<<C, BLOCK_SIZE, 0, stream>>>(input, mean, inv_std, running_mean,
                                                         running_var, N, C, S, momentum, epsilon);

    size_t total_elements = N * C * S;
    size_t total_vectors = total_elements / vec_size;
    int num_blocks = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_apply_kernel_vec<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, mean, inv_std, gamma, beta, output, norm_cache, N, C, S, affine);
  } else {
    fused_stats_kernel<<<C, BLOCK_SIZE, 0, stream>>>(input, mean, inv_std, running_mean,
                                                     running_var, N, C, S, momentum, epsilon);

    size_t total_elements = N * C * S;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_apply_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, mean, inv_std, gamma, beta,
                                                              output, norm_cache, N, C, S, affine);
  }
}

template <typename T>
void run_backward_fused(const T *grad_output, const float *norm_input, const float *inv_std,
                        const float *gamma, float *d_gamma, float *d_beta, T *grad_input, size_t N,
                        size_t C, size_t S, bool affine, cudaStream_t stream) {
  constexpr int vec_size = VectorType<T>::size;
  if (S % vec_size == 0) {
    fused_backward_reduce_kernel_vec<<<C, BLOCK_SIZE, 0, stream>>>(grad_output, norm_input, d_gamma,
                                                                   d_beta, N, C, S);

    size_t total_elements = N * C * S;
    size_t total_vectors = total_elements / vec_size;
    int num_blocks = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_backward_apply_kernel_vec<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        grad_output, norm_input, inv_std, gamma, d_gamma, d_beta, grad_input, N, C, S, affine);
  } else {
    fused_backward_reduce_kernel<<<C, BLOCK_SIZE, 0, stream>>>(grad_output, norm_input, d_gamma,
                                                               d_beta, N, C, S);

    size_t total_elements = N * C * S;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_backward_apply_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        grad_output, norm_input, inv_std, gamma, d_gamma, d_beta, grad_input, N, C, S, affine);
  }
}

template <typename T>
void compute_inference_output(const T *input_data, const float *running_mean_data,
                              const float *running_var_data, const float *gamma_data,
                              const float *beta_data, T *output_data, size_t batch_size,
                              size_t channels, size_t spatial_size, float epsilon, bool affine,
                              cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = BLOCK_SIZE;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  size_t shared_mem_size = 4 * channels * sizeof(float);

  compute_inference_output_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
      input_data, running_mean_data, running_var_data, gamma_data, beta_data, output_data,
      batch_size, channels, spatial_size, epsilon, affine);
}

#define INSTANTIATE_BATCHNORM(T)                                                                   \
  template void compute_inference_output<T>(                                                       \
      const T *input_data, const float *running_mean_data, const float *running_var_data,          \
      const float *gamma_data, const float *beta_data, T *output_data, size_t batch_size,          \
      size_t channels, size_t spatial_size, float epsilon, bool affine, cudaStream_t stream);      \
                                                                                                   \
  template void run_forward_fused<T>(                                                              \
      const T *input, float *mean, float *inv_std, float *running_mean, float *running_var,        \
      const float *gamma, const float *beta, T *output, float *norm_cache, size_t N, size_t C,     \
      size_t S, float momentum, float epsilon, bool affine, cudaStream_t stream);                  \
                                                                                                   \
  template void run_backward_fused<T>(const T *grad_output, const float *norm_input,               \
                                      const float *inv_std, const float *gamma, float *d_gamma,    \
                                      float *d_beta, T *grad_input, size_t N, size_t C, size_t S,  \
                                      bool affine, cudaStream_t stream);

INSTANTIATE_BATCHNORM(fp16)
INSTANTIATE_BATCHNORM(float)
INSTANTIATE_BATCHNORM(double)
#undef INSTANTIATE_BATCHNORM

} // namespace batchnorm_nchw
} // namespace cuda
} // namespace tnn