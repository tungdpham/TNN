/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/sdpa_ops.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <stdexcept>

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 32;

template <typename T>
__global__ void compute_attention_scores_kernel(const T* q, const T* k, T* scores, size_t seq_len,
                                                size_t head_dim, float scale, bool is_causal,
                                                size_t batch_heads_idx) {
  int i = blockIdx.x;
  int j = blockIdx.y;
  int d = threadIdx.x;

  if (d < head_dim) {
    const T* q_ptr = q + batch_heads_idx * seq_len * head_dim + i * head_dim;
    const T* k_ptr = k + batch_heads_idx * seq_len * head_dim + j * head_dim;

    T val = q_ptr[d] * k_ptr[d];
    __syncthreads();

    extern __shared__ char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);
    shared[d] = val;
    __syncthreads();

    for (int stride = head_dim / 2; stride > 0; stride /= 2) {
      if (d < stride) {
        shared[d] += shared[d + stride];
      }
      __syncthreads();
    }

    if (d == 0) {
      T score = shared[0] * scale;
      if (is_causal && j > i) {
        score = -1e9f;
      }
      scores[batch_heads_idx * seq_len * seq_len + i * seq_len + j] = score;
    }
  }
}

template <typename T>
__global__ void softmax_kernel(T* scores, T* attn_weights, size_t seq_len, size_t batch_heads_idx) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  if (j < seq_len) {
    T* score_ptr = scores + batch_heads_idx * seq_len * seq_len + i * seq_len;

    T max_val = -1e9f;
    for (int k = 0; k < seq_len; ++k) {
      max_val = max(max_val, score_ptr[k]);
    }
    __syncthreads();

    T exp_val = exp(score_ptr[j] - max_val);
    T sum_exp = 0;
    for (int k = 0; k < seq_len; ++k) {
      if (k == j) {
        sum_exp += exp_val;
      } else {
        sum_exp += exp(score_ptr[k] - max_val);
      }
    }
    __syncthreads();

    attn_weights[batch_heads_idx * seq_len * seq_len + i * seq_len + j] =
        exp_val / (sum_exp + 1e-9f);
  }
}

template <typename T>
__global__ void attention_output_kernel(const T* attn_weights, const T* v, T* output,
                                        size_t seq_len, size_t head_dim, size_t batch_heads_idx) {
  int i = blockIdx.x;
  int d = threadIdx.x;

  if (d < head_dim) {
    T val = 0;
    for (int j = 0; j < seq_len; ++j) {
      T attn_val = attn_weights[batch_heads_idx * seq_len * seq_len + i * seq_len + j];
      T v_val = v[batch_heads_idx * seq_len * head_dim + j * head_dim + d];
      val += attn_val * v_val;
    }
    output[batch_heads_idx * seq_len * head_dim + i * head_dim + d] = val;
  }
}

template <typename T>
void run_forward(const T* q, const T* k, const T* v, T* output, size_t batch_size, size_t num_heads,
                 size_t seq_len, size_t head_dim, float attn_scale, bool is_causal) {
  size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(T);
  size_t attn_weights_size = batch_size * num_heads * seq_len * seq_len * sizeof(T);

  T* scores;
  T* attn_weights;
  cudaMalloc(&scores, scores_size);
  cudaMalloc(&attn_weights, attn_weights_size);

  if (!scores || !attn_weights) {
    if (scores) cudaFree(scores);
    if (attn_weights) cudaFree(attn_weights);
    throw std::runtime_error("Failed to allocate GPU memory for SDPA forward");
  }

  try {
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        size_t batch_heads_idx = b * num_heads + h;

        dim3 grid(seq_len, seq_len);
        dim3 block(min((int)head_dim, 512));
        size_t shared_mem = head_dim * sizeof(T);
        compute_attention_scores_kernel<T><<<grid, block, shared_mem>>>(
            q, k, scores, seq_len, head_dim, attn_scale, is_causal, batch_heads_idx);

        dim3 softmax_grid(seq_len);
        dim3 softmax_block(seq_len);
        softmax_kernel<T>
            <<<softmax_grid, softmax_block>>>(scores, attn_weights, seq_len, batch_heads_idx);

        dim3 output_grid(seq_len);
        dim3 output_block(min((int)head_dim, 512));
        attention_output_kernel<T><<<output_grid, output_block>>>(attn_weights, v, output, seq_len,
                                                                  head_dim, batch_heads_idx);
      }
    }

    cudaDeviceSynchronize();
  } catch (...) {
    cudaFree(scores);
    cudaFree(attn_weights);
    throw;
  }

  cudaFree(scores);
  cudaFree(attn_weights);
}

template <typename T>
void run_backward(const T* q, const T* k, const T* v, const T* output, const T* grad_output,
                  T* grad_q, T* grad_k, T* grad_v, size_t batch_size, size_t num_heads,
                  size_t seq_len, size_t head_dim, float attn_scale, bool is_causal) {
  size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(T);
  size_t attn_weights_size = batch_size * num_heads * seq_len * seq_len * sizeof(T);
  size_t grad_scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(T);

  T* scores;
  T* attn_weights;
  T* grad_scores;

  cudaMalloc(&scores, scores_size);
  cudaMalloc(&attn_weights, attn_weights_size);
  cudaMalloc(&grad_scores, grad_scores_size);

  if (!scores || !attn_weights || !grad_scores) {
    if (scores) cudaFree(scores);
    if (attn_weights) cudaFree(attn_weights);
    if (grad_scores) cudaFree(grad_scores);
    throw std::runtime_error("Failed to allocate GPU memory for SDPA backward");
  }

  try {
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        size_t batch_heads_idx = b * num_heads + h;

        dim3 grid(seq_len, seq_len);
        dim3 block(min((int)head_dim, 512));
        size_t shared_mem = head_dim * sizeof(T);
        compute_attention_scores_kernel<T><<<grid, block, shared_mem>>>(
            q, k, scores, seq_len, head_dim, attn_scale, is_causal, batch_heads_idx);

        dim3 softmax_grid(seq_len);
        dim3 softmax_block(seq_len);
        softmax_kernel<T>
            <<<softmax_grid, softmax_block>>>(scores, attn_weights, seq_len, batch_heads_idx);
      }
    }

    cudaDeviceSynchronize();

  } catch (...) {
    cudaFree(scores);
    cudaFree(attn_weights);
    cudaFree(grad_scores);
    throw;
  }

  cudaFree(scores);
  cudaFree(attn_weights);
  cudaFree(grad_scores);
}

template void run_forward<float>(const float*, const float*, const float*, float*, size_t, size_t,
                                 size_t, size_t, float, bool);
template void run_forward<double>(const double*, const double*, const double*, double*, size_t,
                                  size_t, size_t, size_t, float, bool);

template void run_backward<float>(const float*, const float*, const float*, const float*,
                                  const float*, float*, float*, float*, size_t, size_t, size_t,
                                  size_t, float, bool);
template void run_backward<double>(const double*, const double*, const double*, const double*,
                                   const double*, double*, double*, double*, size_t, size_t, size_t,
                                   size_t, float, bool);

}  // namespace cuda
}  // namespace tnn

#endif
