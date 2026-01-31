#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {
namespace embedding {

template <typename T>
void compute_embedding_forward(const T *input_data, const T *weight_data, T *output_data,
                               size_t num_indices, size_t vocab_size, size_t embed_dim,
                               size_t padding_idx, cudaStream_t stream);

template <typename T>
void compute_embedding_backward(const T *input_data, const T *gradient_data, T *weight_grad_data,
                                size_t num_indices, size_t vocab_size, size_t embed_dim,
                                size_t padding_idx, cudaStream_t stream);

}  // namespace embedding
}  // namespace cuda
}  // namespace tnn

#endif