/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cuda_runtime.h>

#include "nn/layers_impl/cuda/embedding_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace embedding {

template <typename T>
__global__ void embedding_forward_kernel(const T* input, const T* weight, T* output,
                                         size_t num_indices, size_t vocab_size, size_t embed_dim,
                                         size_t padding_idx) {
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
void compute_embedding_forward(const T* input_data, const T* weight_data, T* output_data,
                               size_t num_indices, size_t vocab_size, size_t embed_dim,
                               size_t padding_idx, cudaStream_t stream) {
  size_t total_elements = num_indices * embed_dim;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  embedding_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
      input_data, weight_data, output_data, num_indices, vocab_size, embed_dim, padding_idx);
}

template <typename T>
__global__ void embedding_backward_kernel(const T* input, const T* grad, T* weight_grad,
                                          size_t num_indices, size_t vocab_size, size_t embed_dim,
                                          size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) return;

  T g_val = grad[tid];
  atomicAdd(&weight_grad[vocab_idx * embed_dim + dim_idx], g_val);
}

template <typename T>
void compute_embedding_backward(const T* input_data, const T* gradient_data, T* weight_grad_data,
                                size_t num_indices, size_t vocab_size, size_t embed_dim,
                                size_t padding_idx, cudaStream_t stream) {
  size_t total_elements = num_indices * embed_dim;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  embedding_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
      input_data, gradient_data, weight_grad_data, num_indices, vocab_size, embed_dim, padding_idx);
}

#define INSTANTIATE_EMBEDDING(T)                                                              \
  template void compute_embedding_forward<T>(const T*, const T*, T*, size_t, size_t, size_t,  \
                                             size_t, cudaStream_t);                           \
                                                                                              \
  template void compute_embedding_backward<T>(const T*, const T*, T*, size_t, size_t, size_t, \
                                              size_t, cudaStream_t);
INSTANTIATE_EMBEDDING(fp16)
INSTANTIATE_EMBEDDING(bf16)
INSTANTIATE_EMBEDDING(float)
INSTANTIATE_EMBEDDING(double)
#undef INSTANTIATE_EMBEDDING

}  // namespace embedding
}  // namespace cuda
}  // namespace tnn
