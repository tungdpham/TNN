/*
 * Copyright (c) 2025 Tung D. Pham
 */
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace tnn {
namespace cuda {

template <typename T>
__global__ void permute_heads_kernel(const T *input, T *output, size_t B, size_t L, size_t H,
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
    output[out_idx] = input[idx];
  }
}

template <typename T>
void permute_heads(const T *input, T *output, size_t B, size_t L, size_t H, size_t D,
                   cudaStream_t stream) {
  size_t total_elements = B * L * H * D;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  permute_heads_kernel<<<numBlocks, blockSize, 0, stream>>>(input, output, B, L, H, D,
                                                            total_elements);
}

template void permute_heads<float>(const float *input, float *output, size_t B, size_t L, size_t H,
                                   size_t D, cudaStream_t stream);
template void permute_heads<double>(const double *input, double *output, size_t B, size_t L,
                                    size_t H, size_t D, cudaStream_t stream);

} // namespace cuda
} // namespace tnn
