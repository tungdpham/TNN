/*
 * Copyright (c) 2025 Tung D. Pham
 */
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#include "type/type.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace tnn {
namespace cuda {

template <typename I_T, typename O_T>
__global__ void permute_heads_kernel(const I_T *input, O_T *output, size_t B, size_t L, size_t H,
                                     size_t D, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    // Calculate b, l, h, d from idx assuming (B, L, H, D) layout
    // idx = b*(L*H*D) + l*(H*D) + h*D + d
    size_t d = idx % D;
    size_t temp = idx / D;
    size_t h = temp % H;
    temp = temp / H;
    size_t l = temp % L;
    size_t b = temp / L;

    // Output layout (B, H, L, D) => out_idx
    size_t out_idx = b * (H * L * D) + h * (L * D) + l * D + d;
    output[out_idx] = static_cast<O_T>(input[idx]);
  }
}

template <typename I_T, typename O_T>
void permute_heads(const I_T *input, O_T *output, size_t B, size_t L, size_t H, size_t D,
                   cudaStream_t stream) {
  size_t total_elements = B * L * H * D;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  permute_heads_kernel<<<numBlocks, blockSize, 0, stream>>>(input, output, B, L, H, D,
                                                            total_elements);
}

#define INSTANTIATE_PERMUTE_HEADS(I_T, O_T)                                                        \
  template void permute_heads<I_T, O_T>(const I_T *input, O_T *output, size_t B, size_t L,         \
                                        size_t H, size_t D, cudaStream_t stream);
#define INSTANTIATE_PERMUTE_HEADS_BOTH(I_T)                                                        \
  INSTANTIATE_PERMUTE_HEADS(I_T, fp16)                                                             \
  INSTANTIATE_PERMUTE_HEADS(I_T, bf16)                                                             \
  INSTANTIATE_PERMUTE_HEADS(I_T, float)                                                            \
  INSTANTIATE_PERMUTE_HEADS(I_T, double)

INSTANTIATE_PERMUTE_HEADS_BOTH(fp16)
INSTANTIATE_PERMUTE_HEADS_BOTH(bf16)
INSTANTIATE_PERMUTE_HEADS_BOTH(float)
INSTANTIATE_PERMUTE_HEADS_BOTH(double)

#undef INSTANTIATE_PERMUTE_HEADS_BOTH
#undef INSTANTIATE_PERMUTE_HEADS

} // namespace cuda
} // namespace tnn
