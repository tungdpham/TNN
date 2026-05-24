/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cuda_runtime.h>

#include <cstdio>

#include "nn/layers_impl/cuda/class_token_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace class_token {
template <typename T>
__global__ void run_forward_imp(const T* input, const T* token, T* output, size_t seq_len,
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
void run_forward(const T* input, const T* token, T* output, size_t batch_size, size_t seq_len,
                 size_t embed_dim, cudaStream_t stream) {
  size_t output_seq_len = seq_len + 1;
  size_t total_elements = batch_size * output_seq_len * embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  run_forward_imp<T>
      <<<blocks, threads, 0, stream>>>(input, token, output, seq_len, embed_dim, total_elements);
}

template <typename T>
__global__ void run_backward_input_imp(const T* grad_output, T* grad_input, size_t seq_len,
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
__global__ void run_backward_token_imp(const T* grad_output, T* grad_token, size_t batch_size,
                                       size_t seq_len, size_t embed_dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_items = batch_size * embed_dim;
  if (idx >= total_items) return;

  size_t e = idx % embed_dim;
  size_t n = idx / embed_dim;

  size_t output_seq_len = seq_len + 1;

  size_t out_idx = n * output_seq_len * embed_dim + 0 * embed_dim + e;

  if constexpr (std::is_floating_point<T>::value) {
    atomicAdd(&grad_token[e], grad_output[out_idx]);
  }
}

template <typename T>
void run_backward(const T* grad_output, T* grad_input, T* grad_token, size_t batch_size,
                  size_t seq_len, size_t embed_dim, cudaStream_t stream) {
  size_t total_input = batch_size * seq_len * embed_dim;
  int threads = 256;
  int blocks = (total_input + threads - 1) / threads;
  run_backward_input_imp<T>
      <<<blocks, threads, 0, stream>>>(grad_output, grad_input, seq_len, embed_dim, total_input);

  size_t total_token_contribs = batch_size * embed_dim;
  blocks = (total_token_contribs + threads - 1) / threads;
  run_backward_token_imp<T>
      <<<blocks, threads, 0, stream>>>(grad_output, grad_token, batch_size, seq_len, embed_dim);
}

#define INSTANTIATE(T)                                                                       \
  template void run_forward<T>(const T* input, const T* token, T* output, size_t batch_size, \
                               size_t seq_len, size_t embed_dim, cudaStream_t stream);       \
  template void run_backward<T>(const T* grad_output, T* grad_input, T* grad_token,          \
                                size_t batch_size, size_t seq_len, size_t embed_dim,         \
                                cudaStream_t stream);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE
}  // namespace class_token
}  // namespace cuda
}  // namespace tnn
