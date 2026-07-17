#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>

#include "cuda/helpers.cuh"
#include "math/cuda/gemm.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "type/cuda/vectorized_types.hpp"
#include "type/type.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__global__ void avgpool_nchw_fwd_kernel(const T* input, T* output, size_t batch_size,
                                        size_t channels, size_t input_h, size_t input_w,
                                        size_t output_h, size_t output_w, size_t pool_h,
                                        size_t pool_w, size_t stride_h, size_t stride_w,
                                        size_t pad_h, size_t pad_w, T pool_size_inv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
  long w_end_valid = min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

  size_t input_offset = (n * channels + c) * input_h * input_w;
  T sum = T(0);

  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      sum += input[input_offset + ih * input_w + iw];
    }
  }

  output[idx] = sum * pool_size_inv;
}

template <typename T>
__global__ void avgpool_nchw_dgrad_kernel(const T* gradient, T* grad_input, size_t batch_size,
                                          size_t channels, size_t input_h, size_t input_w,
                                          size_t output_h, size_t output_w, size_t pool_h,
                                          size_t pool_w, size_t stride_h, size_t stride_w,
                                          size_t pad_h, size_t pad_w, T pool_size_inv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  const T grad_val = gradient[idx] * pool_size_inv;

  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
  long w_end_valid = min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

  size_t input_offset = (n * channels + c) * input_h * input_w;

  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      gpu_atomic_add(&grad_input[input_offset + ih * input_w + iw], grad_val);
    }
  }
}

template <typename T>
void avgpool_nchw_fwd(const T* input, T* output, size_t batch_size, size_t channels, size_t input_h,
                      size_t input_w, size_t output_h, size_t output_w, size_t pool_h,
                      size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                      cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  avgpool_nchw_fwd_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input, output, batch_size, channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
      stride_h, stride_w, pad_h, pad_w, pool_size_inv);
}

template <typename T>
void avgpool_nchw_bwd(const T* gradient, T* grad_input, size_t batch_size, size_t channels,
                      size_t input_h, size_t input_w, size_t output_h, size_t output_w,
                      size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  avgpool_nchw_dgrad_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient, grad_input, batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, pad_h, pad_w, pool_size_inv);
}

template <typename T>
__global__ void avgpool_fwd_kernel(const T* input, T* output, size_t batch_size, size_t height,
                                   size_t width, size_t channels, size_t pool_h, size_t pool_w,
                                   size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                   size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  float sum = 0.0f;
  int count = 0;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      sum += static_cast<float>(input[input_idx]);
      ++count;
    }
  }

  output[idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
}

template <typename T>
__global__ void avgpool_dgrad_kernel(const T* grad_output, T* grad_input, size_t batch_size,
                                     size_t input_h, size_t input_w, size_t channels, size_t pool_h,
                                     size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                     size_t pad_w, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  float grad = static_cast<float>(grad_output[idx]);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  int count = (h_end - h_start) * (w_end - w_start);
  if (count == 0) return;

  float grad_per_element = grad / count;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
      cuda::gpu_atomic_add(&grad_input[input_idx], static_cast<T>(grad_per_element));
    }
  }
}

template <typename T>
struct WelfordData {
  T mean;
  T m2;
  T count;

  __device__ WelfordData()
      : mean(0),
        m2(0),
        count(0) {}
  __device__ WelfordData(T m, T v, T c)
      : mean(m),
        m2(v),
        count(c) {}
};

template <typename T>
__device__ WelfordData<T> welford_merge(WelfordData<T> a, WelfordData<T> b) {
  if (b.count == T(0)) return a;
  if (a.count == T(0)) return b;

  T new_count = a.count + b.count;
  T delta = b.mean - a.mean;
  T new_mean = a.mean + (delta * b.count) / new_count;
  T new_m2 = a.m2 + b.m2 + (delta * delta * a.count * b.count) / new_count;

  return WelfordData<T>(new_mean, new_m2, new_count);
}

template <typename T>
__device__ WelfordData<T> warpReduceWelford(WelfordData<T> val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other_mean = __shfl_down_sync(0xffffffff, val.mean, offset);
    T other_m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
    T other_count = __shfl_down_sync(0xffffffff, val.count, offset);
    val = welford_merge(val, WelfordData<T>(other_mean, other_m2, other_count));
  }
  return val;
}

template <typename T>
__device__ WelfordData<T> blockReduceWelford(WelfordData<T> val) {
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

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
  if (wid == 0) val = warpReduceSum(val);

  return val;
}

template <typename T>
__global__ void batchnorm_stats_kernel(const T* __restrict__ input, float* __restrict__ mean_out,
                                       float* __restrict__ inv_std_out,
                                       float* __restrict__ running_mean,
                                       float* __restrict__ running_var, size_t N, size_t C,
                                       size_t S, float momentum, float epsilon) {
  int c = blockIdx.x;
  if (c >= C) return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;

  WelfordData<float> thread;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * channel_stride + channel_offset + s;

    float val = static_cast<float>(input[idx]);

    thread.count += 1.0f;
    float delta = val - thread.mean;
    thread.mean += delta / thread.count;
    float delta2 = val - thread.mean;
    thread.m2 += delta * delta2;
  }

  WelfordData<float> result = blockReduceWelford(thread);

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
__global__ void batchnorm_stats_kernel_vec(const T* __restrict__ input,
                                           float* __restrict__ mean_out,
                                           float* __restrict__ inv_std_out,
                                           float* __restrict__ running_mean,
                                           float* __restrict__ running_var, size_t N, size_t C,
                                           size_t S, float momentum, float epsilon) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  int c = blockIdx.x;
  if (c >= C) return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  WelfordData<float> thread;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * channel_stride + channel_offset + s;

    VecT val_vec = *reinterpret_cast<const VecT*>(&input[idx]);
    const T* val_arr = reinterpret_cast<const T*>(&val_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float val = static_cast<float>(val_arr[k]);
      thread.count += 1.0f;
      float delta = val - thread.mean;
      thread.mean += delta / thread.count;
      float delta2 = val - thread.mean;
      thread.m2 += delta * delta2;
    }
  }

  WelfordData<float> result = blockReduceWelford(thread);

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
__global__ void batchnorm_fwd_kernel(const T* __restrict__ input, const float* __restrict__ mean,
                                     const float* __restrict__ inv_std,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta, T* __restrict__ output,
                                     float* __restrict__ normalized_cache, size_t N, size_t C,
                                     size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    float mu = mean[c];
    float istd = inv_std[c];
    float x = static_cast<float>(input[idx]);

    float norm = (x - mu) * istd;

    if (normalized_cache) normalized_cache[idx] = norm;

    float res = norm;
    if (affine) {
      res = res * gamma[c] + beta[c];
    }
    output[idx] = static_cast<T>(res);
  }
}

template <typename T>
__global__ void batchnorm_fwd_kernel_vec(
    const T* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ inv_std,
    const float* __restrict__ gamma, const float* __restrict__ beta, T* __restrict__ output,
    float* __restrict__ normalized_cache, size_t N, size_t C, size_t S, bool affine) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    float mu = mean[c];
    float istd = inv_std[c];
    float g = (affine && gamma) ? gamma[c] : 1.0f;
    float b = (affine && beta) ? beta[c] : 0.0f;

    VecT x_vec = reinterpret_cast<const VecT*>(input)[idx];
    const T* x_arr = reinterpret_cast<const T*>(&x_vec);

    VecT out_vec;
    T* out_arr = reinterpret_cast<T*>(&out_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float x = static_cast<float>(x_arr[k]);
      float norm = (x - mu) * istd;
      if (normalized_cache) normalized_cache[scalar_idx + k] = norm;

      float res = norm;
      if (affine) {
        res = res * g + b;
      }
      out_arr[k] = static_cast<T>(res);
    }

    reinterpret_cast<VecT*>(output)[idx] = out_vec;
  }
}

template <typename T>
__global__ void batchnorm_wgrad_bgrad_reduce_kernel(const T* __restrict__ grad_output,
                                                    const float* __restrict__ normalized_input,
                                                    float* __restrict__ d_gamma,
                                                    float* __restrict__ d_beta, size_t N, size_t C,
                                                    size_t S) {
  int c = blockIdx.x;
  if (c >= C) return;

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
__global__ void batchnorm_wgrad_bgrad_reduce_kernel_vec(const T* __restrict__ grad_output,
                                                        const float* __restrict__ normalized_input,
                                                        float* __restrict__ d_gamma,
                                                        float* __restrict__ d_beta, size_t N,
                                                        size_t C, size_t S) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  int c = blockIdx.x;
  if (c >= C) return;

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

    VecT dy_vec = *reinterpret_cast<const VecT*>(&grad_output[idx]);
    const T* dy_arr = reinterpret_cast<const T*>(&dy_vec);

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
__global__ void batchnorm_dgrad_kernel(const T* __restrict__ grad_output,
                                       const float* __restrict__ normalized_input,
                                       const float* __restrict__ inv_std,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ d_gamma,
                                       const float* __restrict__ d_beta, T* __restrict__ grad_input,
                                       size_t N, size_t C, size_t S, bool affine) {
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
__global__ void batchnorm_dgrad_kernel_vec(
    const T* __restrict__ grad_output, const float* __restrict__ normalized_input,
    const float* __restrict__ inv_std, const float* __restrict__ gamma,
    const float* __restrict__ d_gamma, const float* __restrict__ d_beta, T* __restrict__ grad_input,
    size_t N, size_t C, size_t S, bool affine) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

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

    VecT dy_vec = reinterpret_cast<const VecT*>(grad_output)[idx];
    const T* dy_arr = reinterpret_cast<const T*>(&dy_vec);

    VecT dx_vec;
    T* dx_arr = reinterpret_cast<T*>(&dx_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float dy = static_cast<float>(dy_arr[k]);
      float x_hat = normalized_input[scalar_idx + k];
      float term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);
      dx_arr[k] = static_cast<T>(term1 * term2);
    }

    reinterpret_cast<VecT*>(grad_input)[idx] = dx_vec;
  }
}

template <typename T>
__global__ void batchnorm_nchw_inf_kernel(const T* input, const float* running_mean,
                                          const float* running_var, const float* gamma,
                                          const float* beta, T* output, size_t batch_size,
                                          size_t channels, size_t spatial_size, float epsilon,
                                          bool affine, bool use_relu) {
  extern __shared__ char shared_mem[];
  float* s_mean = reinterpret_cast<float*>(shared_mem);
  float* s_inv_std = s_mean + channels;
  float* s_gamma = s_inv_std + channels;
  float* s_beta = s_gamma + channels;

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    s_mean[c] = running_mean[c];
    float var_val = running_var[c];
    s_inv_std[c] = rsqrt(var_val + epsilon);
    if (affine) {
      s_gamma[c] = gamma[c];
      s_beta[c] = beta[c];
    }
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements) return;

  int c = (idx / spatial_size) % channels;

  float input_val = static_cast<float>(input[idx]);
  float normalized_val = (input_val - s_mean[c]) * s_inv_std[c];

  float out_val;
  if (affine) {
    out_val = s_gamma[c] * normalized_val + s_beta[c];
  } else {
    out_val = normalized_val;
  }

  if (use_relu) {
    out_val = out_val > 0.0f ? out_val : 0.0f;
  }

  output[idx] = static_cast<T>(out_val);
}

template <typename T>
__global__ void class_token_fwd_kernel(const T* input, const T* token, T* output, size_t seq_len,
                                       size_t embed_dim, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t output_seq_len = seq_len + 1;

  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_out = tmp % output_seq_len;
  size_t n = tmp / output_seq_len;

  if (s_out == 0) {
    output[idx] = token[e];
  } else {
    size_t s_in = s_out - 1;
    size_t in_idx = n * seq_len * embed_dim + s_in * embed_dim + e;
    output[idx] = input[in_idx];
  }
}

template <typename T>
__global__ void class_token_dgrad_kernel(const T* grad_output, T* grad_input, size_t seq_len,
                                         size_t embed_dim, size_t total_input_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_input_elements) return;

  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_in = tmp % seq_len;
  size_t n = tmp / seq_len;

  size_t output_seq_len = seq_len + 1;

  size_t out_idx = n * output_seq_len * embed_dim + (s_in + 1) * embed_dim + e;

  grad_input[idx] = grad_output[out_idx];
}

template <typename T>
__global__ void class_token_wgrad_kernel(const T* grad_output, T* grad_token, size_t batch_size,
                                         size_t seq_len, size_t embed_dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_items = batch_size * embed_dim;
  if (idx >= total_items) return;

  size_t e = idx % embed_dim;
  size_t n = idx / embed_dim;

  size_t output_seq_len = seq_len + 1;

  size_t out_idx = n * output_seq_len * embed_dim + 0 * embed_dim + e;

  if constexpr (tunx::is_floating<T>::value) {
    cuda::gpu_atomic_add(&grad_token[e], grad_output[out_idx]);
  }
}

template <typename T>
__global__ void conv2d_add_bias_kernel(T* output, const T* bias, size_t batch_size, size_t output_h,
                                       size_t output_w, size_t out_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * out_channels * output_h * output_w;

  if (idx >= total_size) return;

  int remaining = idx % (out_channels * output_h * output_w);
  int c = remaining / (output_h * output_w);

  output[idx] += bias[c];
}

template <typename T>
__global__ void conv2d_nchw_bgrad_kernel(const T* gradient, T* grad_bias, size_t batch_size,
                                         size_t output_h, size_t output_w, size_t out_channels) {
  size_t spatial_size = output_h * output_w;
  size_t channel_stride = spatial_size;
  size_t batch_stride = out_channels * spatial_size;

  int c = blockIdx.x;
  if (c >= out_channels) return;

  extern __shared__ char shared_mem[];
  T* shared = reinterpret_cast<T*>(shared_mem);

  T sum = T(0);

  int tid = threadIdx.x;
  int total_elements = batch_size * spatial_size;

  for (int idx = tid; idx < total_elements; idx += blockDim.x) {
    int n = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    sum += gradient[n * batch_stride + c * channel_stride + spatial_idx];
  }

  shared[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    grad_bias[c] = shared[0];
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void dense_bgrad_kernel_ex(const IO_T* current_grad, Param_T* grad_bias,
                                      size_t batch_size, size_t output_features) {
  int out_f = blockIdx.x;
  if (out_f >= static_cast<int>(output_features)) return;

  extern __shared__ char shared_mem[];
  Compute_T* shared = reinterpret_cast<Compute_T*>(shared_mem);

  Compute_T sum = Compute_T(0);
  int tid = threadIdx.x;
  for (int n = tid; n < static_cast<int>(batch_size); n += blockDim.x) {
    sum += static_cast<Compute_T>(current_grad[n * output_features + out_f]);
  }

  shared[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    Compute_T prev = static_cast<Compute_T>(grad_bias[out_f]);
    grad_bias[out_f] = static_cast<Param_T>(prev + shared[0]);
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void dense_add_bias_kernel(IO_T* output, const Param_T* bias, size_t batch_size,
                                      size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size) return;

  int out_f = idx % output_features;
  output[idx] += static_cast<IO_T>(bias[out_f]);
}

template <typename T>
__global__ void dropout_fwd_kernel_vectorized(const T* __restrict__ input, T* __restrict__ output,
                                              bool* __restrict__ mask, size_t n_elements,
                                              T dropout_rate, T scale, unsigned long long seed) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_width = VectoredTraits<T>::size;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  size_t n_vectors = n_elements / vec_width;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < n_vectors; i += stride) {
    VecT in_val = input_vec[i];
    VecT out_val;
    const T* in_arr = reinterpret_cast<const T*>(&in_val);
    T* out_arr = reinterpret_cast<T*>(&out_val);

    float4 rand_vals = curand_uniform4(&state);
    float rands[4] = {rand_vals.x, rand_vals.y, rand_vals.z, rand_vals.w};

    size_t base = i * vec_width;

#pragma unroll
    for (int k = 0; k < vec_width; ++k) {
      if (rands[k] < static_cast<float>(dropout_rate)) {
        mask[base + k] = false;
        out_arr[k] = T(0);
      } else {
        mask[base + k] = true;
        out_arr[k] = in_arr[k] * scale;
      }
    }

    output_vec[i] = out_val;
  }

  size_t tail_start = n_vectors * vec_width;
  for (size_t i = tail_start + idx; i < n_elements; i += stride) {
    float r = curand_uniform(&state);
    if (r < static_cast<float>(dropout_rate)) {
      mask[i] = false;
      output[i] = T(0);
    } else {
      mask[i] = true;
      output[i] = input[i] * scale;
    }
  }
}

template <typename T>
__global__ void dropout_fwd_kernel(const T* input, T* output, bool* mask, size_t batch_size,
                                   size_t channels, size_t spatial_size, T dropout_rate, T scale,
                                   unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_size * channels * spatial_size;
  int stride = blockDim.x * gridDim.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < total_elements; i += stride) {
    float rand_val = curand_uniform(&state);

    if (rand_val < static_cast<float>(dropout_rate)) {
      mask[i] = false;
      output[i] = T(0);
    } else {
      mask[i] = true;
      output[i] = input[i] * scale;
    }
  }
}

template <typename T>
__global__ void dropout_dgrad_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                     const bool* __restrict__ mask, size_t total_elements,
                                     T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total_elements; i += stride) {
    grad_input[i] = mask[i] ? grad_output[i] * scale : T(0);
  }
}

template <typename T>
__global__ void embedding_fwd_kernel(const T* input, const T* weight, T* output, size_t num_indices,
                                     size_t vocab_size, size_t embed_dim, size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) {
    output[tid] = T(0);
    return;
  }

  output[tid] = weight[vocab_idx * embed_dim + dim_idx];
}

template <typename T>
__global__ void embedding_wgrad_kernel(const T* input, const T* grad, T* grad_weight,
                                       size_t num_indices, size_t vocab_size, size_t embed_dim,
                                       size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) return;

  if constexpr (tunx::is_floating<T>::value) {
    T g_val = grad[tid];
    cuda::gpu_atomic_add(&grad_weight[vocab_idx * embed_dim + dim_idx], g_val);
  }
}

template <typename T>
__global__ void layernorm_fwd_kernel(const T* input, T* output, const T* gamma, const T* beta,
                                     size_t channels, T epsilon) {
  size_t n = static_cast<size_t>(blockIdx.x);

  const T* x = input + n * channels;
  T* y = output + n * channels;

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += x[c];
  }
  const T mean = sum / static_cast<T>(channels);

  T sq_sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    const T diff = x[c] - mean;
    sq_sum += diff * diff;
  }
  const T var = sq_sum / static_cast<T>(channels);
  const T inv_std = T(1) / static_cast<T>(sqrt(static_cast<double>(var + epsilon)));

  for (size_t c = 0; c < channels; ++c) {
    const T norm = (x[c] - mean) * inv_std;
    const T g = gamma ? gamma[c] : T(1);
    const T b = beta ? beta[c] : T(0);
    y[c] = g * norm + b;
  }
}

template <typename T>
__global__ void layernorm_bwd_kernel(const T* grad_output, const T* input, const T* gamma,
                                     T* grad_input, T* grad_gamma, T* grad_beta, size_t channels,
                                     T epsilon) {
  size_t n = static_cast<size_t>(blockIdx.x);

  const T* x = input + n * channels;
  const T* go = grad_output + n * channels;
  T* gi = grad_input ? (grad_input + n * channels) : nullptr;

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += x[c];
  }
  const T mean = sum / static_cast<T>(channels);

  T sq_sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    const T diff = x[c] - mean;
    sq_sum += diff * diff;
  }
  const T var = sq_sum / static_cast<T>(channels);
  const T inv_std = T(1) / static_cast<T>(sqrt(static_cast<double>(var + epsilon)));

  T sum_dl_dnorm = T(0);
  T sum_dl_dnorm_x_hat = T(0);

  for (size_t c = 0; c < channels; ++c) {
    const T g = gamma ? gamma[c] : T(1);
    const T dl_dnorm = go[c] * g;
    const T x_hat = (x[c] - mean) * inv_std;
    sum_dl_dnorm += dl_dnorm;
    sum_dl_dnorm_x_hat += dl_dnorm * x_hat;

    if constexpr (tunx::is_floating<T>::value) {
      if (grad_gamma) {
        cuda::gpu_atomic_add(&grad_gamma[c], go[c] * x_hat);
      }
      if (grad_beta) {
        cuda::gpu_atomic_add(&grad_beta[c], go[c]);
      }
    }
  }

  if (gi) {
    const T mean_dl_dnorm = sum_dl_dnorm / static_cast<T>(channels);
    const T mean_dl_dnorm_x_hat = sum_dl_dnorm_x_hat / static_cast<T>(channels);

    for (size_t c = 0; c < channels; ++c) {
      const T g = gamma ? gamma[c] : T(1);
      const T dl_dnorm = go[c] * g;
      const T x_hat = (x[c] - mean) * inv_std;
      gi[c] = inv_std * (dl_dnorm - mean_dl_dnorm - x_hat * mean_dl_dnorm_x_hat);
    }
  }
}

template <typename T>
__global__ void maxpool_nchw_fwd_kernel(const T* input, T* output, size_t batch_size,
                                        size_t channels, size_t input_h, size_t input_w,
                                        size_t output_h, size_t output_w, size_t pool_h,
                                        size_t pool_w, size_t stride_h, size_t stride_w,
                                        size_t pad_h, size_t pad_w, size_t* mask_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  size_t input_offset = (n * channels + c) * input_h * input_w;

  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);
  long h_end = h_start + pool_h;
  long w_end = w_start + pool_w;

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_end);
  long w_end_valid = min(static_cast<long>(input_w), w_end);

  T max_val = -INFINITY;
  size_t max_idx = 0;

  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      size_t cur_input_idx = input_offset + ih * input_w + iw;
      T val = input[cur_input_idx];

      if (val > max_val || (ih == h_start_valid && iw == w_start_valid)) {
        max_val = val;
        max_idx = cur_input_idx;
      }
    }
  }

  output[idx] = max_val;
  mask_indices[idx] = max_idx;
}

template <typename T>
__global__ void maxpool_nchw_dgrad_kernel(const T* gradient, T* grad_input, size_t batch_size,
                                          size_t channels, size_t output_h, size_t output_w,
                                          const size_t* mask_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  const T grad_val = gradient[idx];
  size_t input_idx = mask_indices[idx];

  gpu_atomic_add(&grad_input[input_idx], grad_val);
}

template <typename T>
__global__ void maxpool_fwd_kernel(const T* input, T* output, int* mask_indices, size_t batch_size,
                                   size_t height, size_t width, size_t channels, size_t pool_h,
                                   size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                   size_t pad_w, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  float max_val = -INFINITY;
  int max_idx = -1;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      float val = static_cast<float>(input[input_idx]);
      if (val > max_val) {
        max_val = val;

        int h_start_unclamped = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
        int w_start_unclamped = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
        int rel_h = h - h_start_unclamped;
        int rel_w = w - w_start_unclamped;
        max_idx = rel_h * static_cast<int>(pool_w) + rel_w;
      }
    }
  }

  output[idx] = static_cast<T>(max_val);
  mask_indices[idx] = max_idx;
}

template <typename T>
__global__ void maxpool_dgrad_kernel(const T* grad_output, T* grad_input, const int* mask_indices,
                                     size_t batch_size, size_t channels, size_t output_h,
                                     size_t output_w, size_t input_h, size_t input_w, size_t pool_w,
                                     size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  int rel_idx = mask_indices[idx];
  if (rel_idx >= 0) {
    size_t c = idx % channels;
    size_t ow = (idx / channels) % output_w;
    size_t oh = (idx / (channels * output_w)) % output_h;
    size_t b = idx / (channels * output_w * output_h);

    int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
    int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

    int rel_h = rel_idx / static_cast<int>(pool_w);
    int rel_w = rel_idx % static_cast<int>(pool_w);

    int h = h_start + rel_h;
    int w = w_start + rel_w;

    if (h >= 0 && h < static_cast<int>(input_h) && w >= 0 && w < static_cast<int>(input_w)) {
      size_t in_idx = ((b * input_h + h) * input_w + w) * channels + c;
      cuda::gpu_atomic_add(&grad_input[in_idx], static_cast<T>(grad_output[idx]));
    }
  }
}

template <typename T>
__global__ void relu_fwd_vec_kernel(const T* input, T* output, uint8_t* mask, size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  VecT in_val = input_vec[idx];
  VecT out_val;

  const T* in_ptr = reinterpret_cast<const T*>(&in_val);
  T* out_ptr = reinterpret_cast<T*>(&out_val);

  T zero = static_cast<T>(0);
  uint8_t mask_vals[VecSize];

  for (int i = 0; i < VecSize; ++i) {
    bool is_positive = in_ptr[i] > zero;
    out_ptr[i] = is_positive ? in_ptr[i] : zero;
    mask_vals[i] = is_positive ? 1 : 0;
  }

  output_vec[idx] = out_val;

  size_t mask_base = idx * VecSize;
  for (int i = 0; i < VecSize; ++i) {
    mask[mask_base + i] = mask_vals[i];
  }
}

template <typename T>
__global__ void relu_fwd_tail_kernel(const T* input, T* output, uint8_t* mask, size_t offset,
                                     size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  T val = input[pos];
  T zero = static_cast<T>(0);
  bool is_positive = val > zero;
  output[pos] = is_positive ? val : zero;
  mask[pos] = is_positive ? 1 : 0;
}

template <typename T>
__global__ void relu_dgrad_vec_kernel(const T* grad_output, T* grad_input, const uint8_t* mask,
                                      size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* grad_out_vec = reinterpret_cast<const VecT*>(grad_output);
  VecT* grad_in_vec = reinterpret_cast<VecT*>(grad_input);

  VecT grad_out_val = grad_out_vec[idx];
  VecT grad_in_val;

  const T* grad_out_ptr = reinterpret_cast<const T*>(&grad_out_val);
  T* grad_in_ptr = reinterpret_cast<T*>(&grad_in_val);

  size_t mask_base = idx * VecSize;

  for (int i = 0; i < VecSize; ++i) {
    uint8_t m = mask[mask_base + i];
    grad_in_ptr[i] = grad_out_ptr[i] * static_cast<T>(m);
  }

  grad_in_vec[idx] = grad_in_val;
}

template <typename T>
__global__ void relu_dgrad_tail_kernel(const T* grad_output, T* grad_input, const uint8_t* mask,
                                       size_t offset, size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  grad_input[pos] = grad_output[pos] * static_cast<T>(mask[pos]);
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = __shfl_down_sync(0xffffffff, val, offset);
    val = val > other ? val : other;
  }
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceMax(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : lowest_value<T>();
  if (wid == 0) {
    val = warpReduceMax(val);
  }

  return val;
}

template <typename T>
__global__ void sdpa_compute_scores_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    typename TypeTraits<T>::ComputePrecision* __restrict__ scores, size_t batch_heads,
    size_t seq_len, size_t head_dim, float scale, bool is_causal) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const T* q_row = q + data_base + query_idx * head_dim;
  AccT* score_row = scores + row * seq_len;

  for (size_t key_idx = threadIdx.x; key_idx < seq_len; key_idx += blockDim.x) {
    if (is_causal && key_idx > query_idx) {
      score_row[key_idx] = lowest_value<AccT>();
      continue;
    }

    const T* k_row = k + data_base + key_idx * head_dim;
    AccT dot = AccT(0);
    for (size_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
      dot += static_cast<AccT>(q_row[dim_idx]) * static_cast<AccT>(k_row[dim_idx]);
    }
    score_row[key_idx] = dot * static_cast<AccT>(scale);
  }
}

template <typename T>
__global__ void sdpa_softmax_kernel(
    const typename TypeTraits<T>::ComputePrecision* __restrict__ scores,
    T* __restrict__ attn_weights, size_t total_rows, size_t seq_len) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  if (row >= total_rows) {
    return;
  }

  const AccT* score_row = scores + row * seq_len;
  T* attn_row = attn_weights + row * seq_len;

  AccT local_max = lowest_value<AccT>();
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    AccT value = score_row[idx];
    local_max = local_max > value ? local_max : value;
  }
  AccT row_max = blockReduceMax(local_max);

  __shared__ AccT shared_max;
  if (threadIdx.x == 0) {
    shared_max = row_max;
  }
  __syncthreads();
  row_max = shared_max;

  AccT local_sum = AccT(0);
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    local_sum += device_exp(static_cast<AccT>(score_row[idx] - row_max));
  }
  AccT row_sum = blockReduceSum(local_sum);

  __shared__ AccT shared_sum;
  if (threadIdx.x == 0) {
    shared_sum = row_sum;
  }
  __syncthreads();
  row_sum = shared_sum;

  AccT inv_sum = AccT(1) / (row_sum + static_cast<AccT>(1e-9f));
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    attn_row[idx] =
        static_cast<T>(device_exp(static_cast<AccT>(score_row[idx] - row_max)) * inv_sum);
  }
}

template <typename T>
__global__ void sdpa_output_kernel(const T* __restrict__ attn_weights, const T* __restrict__ v,
                                   T* __restrict__ output, size_t batch_heads, size_t seq_len,
                                   size_t head_dim) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const T* attn_row = attn_weights + row * seq_len;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t key_idx = 0; key_idx < seq_len; ++key_idx) {
      value += static_cast<AccT>(attn_row[key_idx]) *
               static_cast<AccT>(v[data_base + key_idx * head_dim + dim_idx]);
    }
    output[data_base + query_idx * head_dim + dim_idx] = static_cast<T>(value);
  }
}

template <typename T>
__global__ void sdpa_dgrad_v_kernel(const T* __restrict__ attn_weights,
                                    const T* __restrict__ grad_output, T* __restrict__ grad_v,
                                    size_t batch_heads, size_t seq_len, size_t head_dim) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t key_row = blockIdx.x;
  size_t total_key_rows = batch_heads * seq_len;
  if (key_row >= total_key_rows) {
    return;
  }

  size_t batch_head = key_row / seq_len;
  size_t key_idx = key_row % seq_len;
  size_t attn_base = batch_head * seq_len * seq_len;
  size_t data_base = batch_head * seq_len * head_dim;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t query_idx = 0; query_idx < seq_len; ++query_idx) {
      value += static_cast<AccT>(attn_weights[attn_base + query_idx * seq_len + key_idx]) *
               static_cast<AccT>(grad_output[data_base + query_idx * head_dim + dim_idx]);
    }
    grad_v[data_base + key_idx * head_dim + dim_idx] = static_cast<T>(value);
  }
}

template <typename T>
__global__ void sdpa_dgrad_attn_kernel(
    const T* __restrict__ grad_output, const T* __restrict__ v,
    typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores, size_t batch_heads,
    size_t seq_len, size_t head_dim) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const T* grad_output_row = grad_output + data_base + query_idx * head_dim;
  AccT* grad_score_row = grad_scores + row * seq_len;

  for (size_t key_idx = threadIdx.x; key_idx < seq_len; key_idx += blockDim.x) {
    const T* v_row = v + data_base + key_idx * head_dim;
    AccT value = AccT(0);
    for (size_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
      value += static_cast<AccT>(grad_output_row[dim_idx]) * static_cast<AccT>(v_row[dim_idx]);
    }
    grad_score_row[key_idx] = value;
  }
}

template <typename T>
__global__ void sdpa_softmax_dgrad_kernel(
    const T* __restrict__ attn_weights,
    typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores, size_t total_rows,
    size_t seq_len) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  if (row >= total_rows) {
    return;
  }

  const T* attn_row = attn_weights + row * seq_len;
  AccT* grad_score_row = grad_scores + row * seq_len;

  AccT local_dot = AccT(0);
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    local_dot += static_cast<AccT>(attn_row[idx]) * grad_score_row[idx];
  }
  AccT dot = blockReduceSum(local_dot);

  __shared__ AccT shared_dot;
  if (threadIdx.x == 0) {
    shared_dot = dot;
  }
  __syncthreads();
  dot = shared_dot;

  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    grad_score_row[idx] = static_cast<AccT>(attn_row[idx]) * (grad_score_row[idx] - dot);
  }
}

template <typename T>
__global__ void sdpa_dgrad_q_kernel(
    const typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores,
    const T* __restrict__ k, T* __restrict__ grad_q, size_t batch_heads, size_t seq_len,
    size_t head_dim, float attn_scale) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const AccT* grad_score_row = grad_scores + row * seq_len;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t key_idx = 0; key_idx < seq_len; ++key_idx) {
      value +=
          grad_score_row[key_idx] * static_cast<AccT>(k[data_base + key_idx * head_dim + dim_idx]);
    }
    grad_q[data_base + query_idx * head_dim + dim_idx] =
        static_cast<T>(value * static_cast<AccT>(attn_scale));
  }
}

template <typename T>
__global__ void sdpa_dgrad_k_kernel(
    const typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores,
    const T* __restrict__ q, T* __restrict__ grad_k, size_t batch_heads, size_t seq_len,
    size_t head_dim, float attn_scale) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t key_row = blockIdx.x;
  size_t total_key_rows = batch_heads * seq_len;
  if (key_row >= total_key_rows) {
    return;
  }

  size_t batch_head = key_row / seq_len;
  size_t key_idx = key_row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const AccT* grad_score_base = grad_scores + batch_head * seq_len * seq_len;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t query_idx = 0; query_idx < seq_len; ++query_idx) {
      value += grad_score_base[query_idx * seq_len + key_idx] *
               static_cast<AccT>(q[data_base + query_idx * head_dim + dim_idx]);
    }
    grad_k[data_base + key_idx * head_dim + dim_idx] =
        static_cast<T>(value * static_cast<AccT>(attn_scale));
  }
}

template <typename T>
__global__ void slice_fwd_kernel(const T* input, T* output, size_t outer_size, size_t inner_size,
                                 size_t axis_size, size_t start, size_t length,
                                 size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t i = idx % inner_size;
  size_t tmp = idx / inner_size;
  size_t l = tmp % length;
  size_t o = tmp / length;

  size_t input_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
  output[idx] = input[input_idx];
}

template <typename T>
__global__ void slice_dgrad_kernel(const T* grad_output, T* grad_input, size_t outer_size,
                                   size_t inner_size, size_t axis_size, size_t start, size_t length,
                                   size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t i = idx % inner_size;
  size_t tmp = idx / inner_size;
  size_t l = tmp % length;
  size_t o = tmp / length;

  size_t output_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
  grad_input[output_idx] = grad_output[idx];
}

void* CUDAEngine::create_backend_handle() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  return stream;
}

WorkspaceReq CUDAEngine::query_dense_graph(void* backend_handle, const DenseStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_avgpool_graph(void* backend_handle, const AvgPool2DStats& stats,
                                             DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_maxpool2d_graph(void* backend_handle, const MaxPool2DStats& stats,
                                               DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_class_token_graph(void* backend_handle, const ClassTokenStats& stats,
                                                 DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_dropout_graph(void* backend_handle, const DropoutStats& stats,
                                             DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_embedding_graph(void* backend_handle, const EmbeddingStats& stats,
                                               DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_relu_graph(void* backend_handle, const ReLUStats& stats,
                                          DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_batchnorm_graph(void* backend_handle, const BatchNormStats& stats,
                                               DTypeDesc type_desc) {
  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CUDAEngine::query_conv2d_graph(void* backend_handle, const Conv2DStats& stats,
                                            DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_layernorm_graph(void* backend_handle, const LayerNormStats& stats,
                                               DTypeDesc type_desc) {
  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

void CUDAEngine::dense_fwd(void* backend_handle, const DenseStats& stats, const void* input,
                           const void* weight, const void* bias, void* output, void* workspace,
                           DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cuda::gemm_ex<T, T, T, float>(static_cast<const T*>(input), static_cast<const T*>(weight),
                                  static_cast<T*>(output), stats.batch_size, stats.out_features,
                                  stats.in_features, false, true, 1.0f, 0.0f, stats.in_features,
                                  stats.in_features, stats.out_features, stream);
    if (stats.use_bias) {
      int total_size = stats.batch_size * stats.out_features;
      int threads_per_block = 256;
      int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
      dense_add_bias_kernel<T, T, float><<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<T*>(output), static_cast<const T*>(bias), stats.batch_size,
          stats.out_features);
    }
  });
}

void CUDAEngine::dense_wgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                             const void* input, void* grad_weight_prev, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cuda::gemm_ex<T, T, T, float>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                                  static_cast<T*>(grad_weight_prev), stats.out_features,
                                  stats.in_features, stats.batch_size, true, false, 1.0f, 1.0f,
                                  stats.out_features, stats.in_features, stats.in_features, stream);
  });
}

void CUDAEngine::dense_dgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                             const void* weight, void* grad_input, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cuda::gemm_ex<T, T, T, float>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                                  static_cast<T*>(grad_input), stats.batch_size, stats.in_features,
                                  stats.out_features, false, false, 1.0f, 0.0f, stats.out_features,
                                  stats.in_features, stats.in_features, stream);
  });
}

void CUDAEngine::dense_bgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                             void* grad_bias_prev, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    int threads_per_block_b = 256;
    int num_blocks_b = static_cast<int>(stats.out_features);
    size_t shared_mem_size = threads_per_block_b * sizeof(float);
    dense_bgrad_kernel_ex<T, T, float>
        <<<num_blocks_b, threads_per_block_b, shared_mem_size, stream>>>(
            static_cast<const T*>(grad_output), static_cast<T*>(grad_bias_prev), stats.batch_size,
            stats.out_features);
  });
}

void CUDAEngine::avgpool_fwd(void* backend_handle, const AvgPool2DStats& stats, const void* input,
                             void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    avgpool_fwd_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), stats.batch_size, stats.height,
        stats.width, stats.channels, stats.pool_h, stats.pool_w, stats.stride_h, stats.stride_w,
        stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::avgpool_bwd(void* backend_handle, const AvgPool2DStats& stats,
                             const void* grad_output, void* grad_input, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    avgpool_dgrad_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), stats.batch_size,
        stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
        stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::maxpool2d_fwd(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                               void* output, void* mask, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    maxpool_fwd_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<int*>(mask),
        stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::maxpool2d_infer(void* backend_handle, const MaxPool2DStats& stats,
                                 const void* input, void* output, void* workspace,
                                 DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    maxpool_fwd_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<int*>(nullptr),
        stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::maxpool2d_bwd(void* backend_handle, const MaxPool2DStats& stats,
                               const void* grad_output, void* grad_input, const void* mask,
                               void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    maxpool_dgrad_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
        static_cast<const int*>(mask), stats.batch_size, stats.channels, output_h, output_w,
        stats.height, stats.width, stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h,
        stats.pad_w);
  });
}

void CUDAEngine::class_token_fwd(void* backend_handle, const ClassTokenStats& stats,
                                 const void* input, const void* token, void* output,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t output_seq_len = stats.seq_len + 1;
    size_t total_elements = stats.batch_size * output_seq_len * stats.embed_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    class_token_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(token), static_cast<T*>(output),
        stats.seq_len, stats.embed_dim, total_elements);
  });
}

void CUDAEngine::class_token_bwd(void* backend_handle, const ClassTokenStats& stats,
                                 const void* grad_output, void* grad_input, void* grad_token_prev,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_input = stats.batch_size * stats.seq_len * stats.embed_dim;
    int threads = 256;
    int blocks = (total_input + threads - 1) / threads;
    class_token_dgrad_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), stats.seq_len,
        stats.embed_dim, total_input);

    size_t total_token_contribs = stats.batch_size * stats.embed_dim;
    blocks = (total_token_contribs + threads - 1) / threads;
    class_token_wgrad_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_token_prev), stats.batch_size,
        stats.seq_len, stats.embed_dim);
  });
}

void CUDAEngine::dropout_fwd(void* backend_handle, const DropoutStats& stats, const void* input,
                             void* output, bool* mask, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 4096);
    T scale = T(1.0f) / (T(1.0f) - static_cast<T>(stats.dropout_rate));
    unsigned long long seed = static_cast<unsigned long long>(clock()) +
                              static_cast<unsigned long long>(std::time(nullptr));

    dropout_fwd_kernel_vectorized<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<bool*>(mask),
        total_elements, static_cast<T>(stats.dropout_rate), scale, seed);
  });
}

void CUDAEngine::dropout_bwd(void* backend_handle, const DropoutStats& stats,
                             const void* grad_output, void* grad_input, const bool* mask,
                             double scale_in, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 4096);
    T scale = T(1.0f) / (T(1.0f) - static_cast<T>(stats.dropout_rate));
    dropout_dgrad_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
        static_cast<const bool*>(mask), total_elements, scale);
  });
}

void CUDAEngine::relu_fwd(void* backend_handle, const ReLUStats& stats, const void* input,
                          void* output, bool* mask, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t num_elements = stats.batch_size * stats.spatial_size;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (num_elements > 0) {
      constexpr int VecSize = VectoredTraits<T>::size;
      constexpr int threads = 256;
      size_t n_vectors = num_elements / VecSize;
      if (n_vectors > 0) {
        int blocks = (n_vectors + threads - 1) / threads;
        relu_fwd_vec_kernel<T>
            <<<blocks, threads, 0, stream>>>(static_cast<const T*>(input), static_cast<T*>(output),
                                             reinterpret_cast<uint8_t*>(mask), n_vectors);
      }
      size_t tail_offset = n_vectors * VecSize;
      size_t tail_elements = num_elements - tail_offset;
      if (tail_elements > 0) {
        int blocks = (tail_elements + threads - 1) / threads;
        relu_fwd_tail_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(output), reinterpret_cast<uint8_t*>(mask),
            tail_offset, tail_elements);
      }
    }
  });
}

void CUDAEngine::relu_bwd(void* backend_handle, const ReLUStats& stats, const void* grad_output,
                          void* grad_input, const bool* mask, void* workspace,
                          DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  size_t num_elements = stats.batch_size * stats.spatial_size;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (num_elements > 0) {
      constexpr int VecSize = VectoredTraits<T>::size;
      constexpr int threads = 256;
      size_t n_vectors = num_elements / VecSize;
      if (n_vectors > 0) {
        int blocks = (n_vectors + threads - 1) / threads;
        relu_dgrad_vec_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
            reinterpret_cast<const uint8_t*>(mask), n_vectors);
      }
      size_t tail_offset = n_vectors * VecSize;
      size_t tail_elements = num_elements - tail_offset;
      if (tail_elements > 0) {
        int blocks = (tail_elements + threads - 1) / threads;
        relu_dgrad_tail_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
            reinterpret_cast<const uint8_t*>(mask), tail_offset, tail_elements);
      }
    }
  });
}

void CUDAEngine::embedding_fwd(void* backend_handle, const EmbeddingStats& stats, const void* input,
                               const void* weight, void* output, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.num_indices * stats.embed_dim;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    embedding_fwd_kernel<<<numBlocks, blockSize, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(weight), static_cast<T*>(output),
        stats.num_indices, stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CUDAEngine::embedding_bwd(void* backend_handle, const EmbeddingStats& stats,
                               const void* grad_output, const void* input, void* grad_weight_prev,
                               void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.num_indices * stats.embed_dim;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    embedding_wgrad_kernel<<<numBlocks, blockSize, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(grad_output),
        static_cast<T*>(grad_weight_prev), stats.num_indices, stats.vocab_size, stats.embed_dim,
        stats.padding_idx);
  });
}

void CUDAEngine::batchnorm_fwd(void* backend_handle, const BatchNormStats& stats, const void* input,
                               const void* gamma, const void* beta, void* output,
                               void* prev_running_mean, void* prev_running_var,
                               void* next_running_mean, void* next_running_var, void* batch_mean,
                               void* batch_invar, void* relu_mask, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    if ((stats.height * stats.width) % vec_size == 0) {
      batchnorm_stats_kernel_vec<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<float*>(next_running_mean),
          static_cast<float*>(next_running_var), stats.batch_size, stats.channels,
          (stats.height * stats.width), stats.momentum, stats.epsilon);
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      size_t total_vectors = total_elements / vec_size;
      int num_blocks = (total_vectors + 256 - 1) / 256;
      batchnorm_fwd_kernel_vec<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<const float*>(beta), static_cast<T*>(output), static_cast<float*>(workspace),
          stats.batch_size, stats.channels, (stats.height * stats.width), true);
    } else {
      batchnorm_stats_kernel<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<float*>(next_running_mean),
          static_cast<float*>(next_running_var), stats.batch_size, stats.channels,
          (stats.height * stats.width), stats.momentum, stats.epsilon);
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      int num_blocks = (total_elements + 256 - 1) / 256;
      batchnorm_fwd_kernel<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<const float*>(beta), static_cast<T*>(output), static_cast<float*>(workspace),
          stats.batch_size, stats.channels, (stats.height * stats.width), true);
    }
  });
}

void CUDAEngine::batchnorm_infer(void* backend_handle, const BatchNormStats& stats,
                                 const void* input, const void* gamma, const void* beta,
                                 const void* saved_mean, const void* saved_var, void* output,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = 4 * stats.channels * sizeof(float);
    batchnorm_nchw_inf_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        static_cast<const T*>(input), static_cast<const float*>(saved_mean),
        static_cast<const float*>(saved_var), static_cast<const float*>(gamma),
        static_cast<const float*>(beta), static_cast<T*>(output), stats.batch_size, stats.channels,
        (stats.height * stats.width), stats.epsilon, true, stats.use_relu);
  });
}

void CUDAEngine::batchnorm_bwd(void* backend_handle, const BatchNormStats& stats,
                               const void* grad_output, const void* input, const void* relu_mask,
                               const void* gamma, void* grad_input, void* grad_gamma,
                               void* grad_beta, const void* batch_mean, const void* batch_invar,
                               void* workspace, DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    if ((stats.height * stats.width) % vec_size == 0) {
      batchnorm_wgrad_bgrad_reduce_kernel_vec<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          stats.batch_size, stats.channels, (stats.height * stats.width));
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      size_t total_vectors = total_elements / vec_size;
      int num_blocks = (total_vectors + 256 - 1) / 256;
      batchnorm_dgrad_kernel_vec<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<const float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          static_cast<T*>(grad_input), stats.batch_size, stats.channels,
          (stats.height * stats.width), true);
    } else {
      batchnorm_wgrad_bgrad_reduce_kernel<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          stats.batch_size, stats.channels, (stats.height * stats.width));
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      int num_blocks = (total_elements + 256 - 1) / 256;
      batchnorm_dgrad_kernel<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<const float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          static_cast<T*>(grad_input), stats.batch_size, stats.channels,
          (stats.height * stats.width), true);
    }

    if (true) {
      int total_size = 1 * stats.channels;
      int num_blocks = (total_size + 256 - 1) / 256;
      dense_add_bias_kernel<float, float, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<float*>(grad_gamma), static_cast<const float*>(grad_gamma_temp), 1,
          stats.channels);
      dense_add_bias_kernel<float, float, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<float*>(grad_beta), static_cast<const float*>(grad_beta_temp), 1,
          stats.channels);
    }
  });
}

void CUDAEngine::conv2d_fwd(void* backend_handle, const Conv2DStats& stats, const void* input,
                            const void* weight, const void* bias, void* output, void* workspace,
                            DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_fwd not implemented");
}

void CUDAEngine::conv2d_dgrad(void* backend_handle, const Conv2DStats& stats,
                              const void* grad_output, const void* weight, void* grad_input,
                              void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_dgrad not implemented");
}

void CUDAEngine::conv2d_wgrad(void* backend_handle, const Conv2DStats& stats,
                              const void* grad_output, const void* input, void* grad_weight_prev,
                              void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_wgrad not implemented");
}

void CUDAEngine::conv2d_bgrad(void* backend_handle, const Conv2DStats& stats,
                              const void* grad_output, void* grad_bias_prev, void* workspace,
                              DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_bgrad not implemented");
}

void CUDAEngine::layernorm_fwd(void* backend_handle, const LayerNormStats& stats, const void* input,
                               const void* gamma, const void* beta, void* output, void* mean,
                               void* inv_variance, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (stats.batch_size == 0 || stats.channels == 0) return;
    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
    dim3 threads(1);
    layernorm_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<const T*>(gamma),
        static_cast<const T*>(beta), stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CUDAEngine::layernorm_infer(void* backend_handle, const LayerNormStats& stats,
                                 const void* input, const void* gamma, const void* beta,
                                 void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (stats.batch_size == 0 || stats.channels == 0) return;
    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
    dim3 threads(1);
    layernorm_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<const T*>(gamma),
        static_cast<const T*>(beta), stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CUDAEngine::layernorm_bwd(void* backend_handle, const LayerNormStats& stats,
                               const void* grad_output, const void* input, const void* gamma,
                               const void* mean, const void* inv_variance, void* grad_input,
                               void* grad_gamma_prev, void* grad_beta_prev, void* workspace,
                               DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (stats.batch_size == 0 || stats.channels == 0) return;
    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
    dim3 threads(1);
    layernorm_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<const T*>(input),
        static_cast<const T*>(gamma), static_cast<T*>(grad_input), static_cast<T*>(grad_gamma_temp),
        static_cast<T*>(grad_beta_temp), stats.channels, static_cast<T>(stats.epsilon));

    if (true) {
      int total_size = 1 * stats.channels;
      int num_blocks = (total_size + 256 - 1) / 256;
      dense_add_bias_kernel<T, T, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<T*>(grad_gamma_prev), static_cast<const T*>(grad_gamma_temp), 1,
          stats.channels);
      dense_add_bias_kernel<T, T, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<T*>(grad_beta_prev), static_cast<const T*>(grad_beta_temp), 1,
          stats.channels);
    }
  });
}

struct CudaTransposeParams {
  size_t ndim;
  size_t dim0;
  size_t dim1;
  size_t shape[8];
  size_t strides[8];
  size_t out_strides[8];
};

template <typename T>
__global__ void transpose_kernel(const T* input, T* output, CudaTransposeParams p, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t in_idx = idx;
  size_t out_idx = 0;
  size_t coords[8];

  for (size_t i = 0; i < p.ndim; ++i) {
    coords[i] = in_idx / p.strides[i];
    in_idx %= p.strides[i];
  }
  
  size_t temp = coords[p.dim0];
  coords[p.dim0] = coords[p.dim1];
  coords[p.dim1] = temp;

  for (size_t i = 0; i < p.ndim; ++i) {
    out_idx += coords[i] * p.out_strides[i];
  }
  
  output[out_idx] = input[idx];
}

WorkspaceReq CUDAEngine::query_sdpa_graph(void* backend_handle, const AttentionStats& stats,
                                          DTypeDesc type_desc) {
  throw std::runtime_error("SDPA is not implemented for generic CUDAEngine. Use CuDNNEngine.");
}

void CUDAEngine::sdpa_fwd(void* backend_handle, const AttentionStats& stats, const void* q_data,
                          const void* k_data, const void* v_data, void* o_data, void* stats_data,
                          void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("SDPA forward is not implemented for generic CUDAEngine. Use CuDNNEngine.");
}

void CUDAEngine::sdpa_bwd(void* backend_handle, const AttentionStats& stats, const void* q_data,
                          const void* k_data, const void* v_data, const void* o_data,
                          const void* dO_data, const void* stats_data, void* dQ_data, void* dK_data,
                          void* dV_data, void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("SDPA backward is not implemented for generic CUDAEngine. Use CuDNNEngine.");
}

WorkspaceReq CUDAEngine::query_transpose_graph(void* backend_handle, const TransposeStats& stats,
                                               DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

void CUDAEngine::transpose(void* backend_handle, const TransposeStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = static_cast<cudaStream_t>(backend_handle);
  
  CudaTransposeParams p;
  p.ndim = stats.ndim;
  p.dim0 = stats.dim0;
  p.dim1 = stats.dim1;
  
  size_t total_elements = 1;
  for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
    p.shape[i] = stats.shape[i];
    p.strides[i] = total_elements;
    total_elements *= p.shape[i];
  }
  
  size_t out_shape[8];
  for(size_t i=0; i<p.ndim; ++i) out_shape[i] = p.shape[i];
  std::swap(out_shape[p.dim0], out_shape[p.dim1]);
  
  size_t out_total = 1;
  for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
    p.out_strides[i] = out_total;
    out_total *= out_shape[i];
  }
  
  if (total_elements == 0) return;
  
  size_t threads = 256;
  size_t blocks = (total_elements + threads - 1) / threads;
  
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    transpose_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), p, total_elements);
  });
}

}  // namespace tunx
