/*
 * Copyright (c) 2025 Tung D. Pham
 */
#include "nn/blocks_impl/cuda/causal_mask.hpp"
#include "type/type.hpp"
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {

template <typename T>
__global__ void fill_causal_mask_kernel(T *mask, size_t batch_count, size_t L, T neg_inf) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_count * L * L;

  if (idx < total_elements) {
    size_t j = idx % L;
    size_t i = (idx / L) % L;

    mask[idx] = (j > i) ? neg_inf : static_cast<T>(0);
  }
}

template <typename T>
void fill_causal_mask(T *mask, size_t batch_count, size_t L, T neg_inf, cudaStream_t stream) {
  size_t total_elements = batch_count * L * L;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  fill_causal_mask_kernel<<<blocks, threads, 0, stream>>>(mask, batch_count, L, neg_inf);
}

template <typename T>
__global__ void apply_causal_mask_kernel(T *scores, size_t batch_count, size_t L, T neg_inf) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_count * L * L;

  if (idx < total_elements) {
    size_t j = idx % L;
    size_t i = (idx / L) % L;

    if (j > i) {
      scores[idx] = neg_inf;
    }
  }
}

template <typename T>
void apply_causal_mask(T *scores, size_t batch_count, size_t L, T neg_inf, cudaStream_t stream) {
  size_t total_elements = batch_count * L * L;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  apply_causal_mask_kernel<<<blocks, threads, 0, stream>>>(scores, batch_count, L, neg_inf);
}

#define INSTANTIATE_APPLY_CAUSAL_MASK(T)                                                           \
  template void fill_causal_mask<T>(T * mask, size_t batch_count, size_t L, T neg_inf,             \
                                    cudaStream_t stream);                                          \
  template void apply_causal_mask<T>(T * scores, size_t batch_count, size_t L, T neg_inf,          \
                                     cudaStream_t stream);
INSTANTIATE_APPLY_CAUSAL_MASK(fp16)
INSTANTIATE_APPLY_CAUSAL_MASK(bf16)
INSTANTIATE_APPLY_CAUSAL_MASK(float)
INSTANTIATE_APPLY_CAUSAL_MASK(double)
#undef INSTANTIATE_APPLY_CAUSAL_MASK

} // namespace cuda
} // namespace tnn
