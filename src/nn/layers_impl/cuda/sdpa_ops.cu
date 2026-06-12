/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/sdpa_ops.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "type/type.hpp"

namespace synet {
namespace cuda {
namespace sdpa {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__device__ __forceinline__ T lowest_value();

template <>
__device__ __forceinline__ float lowest_value<float>() {
  return -1.0e20f;
}

template <>
__device__ __forceinline__ double lowest_value<double>() {
  return -1.0e300;
}

template <typename T>
__device__ __forceinline__ T device_exp(T value);

template <>
__device__ __forceinline__ float device_exp<float>(float value) {
  return expf(value);
}

template <>
__device__ __forceinline__ double device_exp<double>(double value) {
  return exp(value);
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
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
  if (wid == 0) {
    val = warpReduceSum(val);
  }

  return val;
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
__global__ void compute_scores_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                      typename TypeTraits<T>::ComputePrecision* __restrict__ scores,
                                      size_t batch_heads, size_t seq_len, size_t head_dim,
                                      float scale, bool is_causal) {
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
__global__ void softmax_kernel(const typename TypeTraits<T>::ComputePrecision* __restrict__ scores,
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
    local_sum += device_exp(score_row[idx] - row_max);
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
    attn_row[idx] = static_cast<T>(device_exp(score_row[idx] - row_max) * inv_sum);
  }
}

template <typename T>
__global__ void attention_output_kernel(const T* __restrict__ attn_weights, const T* __restrict__ v,
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
__global__ void grad_v_kernel(const T* __restrict__ attn_weights, const T* __restrict__ grad_output,
                              T* __restrict__ grad_v, size_t batch_heads, size_t seq_len,
                              size_t head_dim) {
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
__global__ void grad_attn_kernel(const T* __restrict__ grad_output, const T* __restrict__ v,
                                 typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores,
                                 size_t batch_heads, size_t seq_len, size_t head_dim) {
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
__global__ void softmax_backward_kernel(
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
__global__ void grad_q_kernel(
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
__global__ void grad_k_kernel(
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
void run_forward(const T* q, const T* k, const T* v, T* output,
                 typename TypeTraits<T>::ComputePrecision* scores, T* attn_weights,
                 size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                 float attn_scale, bool is_causal, cudaStream_t stream) {
  size_t batch_heads = batch_size * num_heads;
  size_t total_rows = batch_heads * seq_len;

  compute_scores_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      q, k, scores, batch_heads, seq_len, head_dim, attn_scale, is_causal);
  softmax_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      scores, attn_weights, total_rows, seq_len);
  attention_output_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      attn_weights, v, output, batch_heads, seq_len, head_dim);
}

template <typename T>
void run_backward(const T* q, const T* k, const T* v, const T* attn_weights, const T* grad_output,
                  typename TypeTraits<T>::ComputePrecision* grad_scores, T* grad_q, T* grad_k,
                  T* grad_v, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                  float attn_scale, bool is_causal, cudaStream_t stream) {
  (void)is_causal;

  size_t batch_heads = batch_size * num_heads;
  size_t total_rows = batch_heads * seq_len;

  grad_v_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      attn_weights, grad_output, grad_v, batch_heads, seq_len, head_dim);
  grad_attn_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      grad_output, v, grad_scores, batch_heads, seq_len, head_dim);
  softmax_backward_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      attn_weights, grad_scores, total_rows, seq_len);
  grad_q_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      grad_scores, k, grad_q, batch_heads, seq_len, head_dim, attn_scale);
  grad_k_kernel<T><<<static_cast<unsigned int>(total_rows), BLOCK_SIZE, 0, stream>>>(
      grad_scores, q, grad_k, batch_heads, seq_len, head_dim, attn_scale);
}

#define INSTANTIATE(T)                                                                         \
  template void run_forward<T>(const T*, const T*, const T*, T*,                               \
                               typename TypeTraits<T>::ComputePrecision*, T*, size_t, size_t,  \
                               size_t, size_t, float, bool, cudaStream_t);                     \
  template void run_backward<T>(const T*, const T*, const T*, const T*, const T*,              \
                                typename TypeTraits<T>::ComputePrecision*, T*, T*, T*, size_t, \
                                size_t, size_t, size_t, float, bool, cudaStream_t);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace sdpa
}  // namespace cuda
}  // namespace synet

#endif
