/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/dropout_ops.hpp"

#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tnn {
namespace cuda {

#define BLOCK_SIZE 256

template <typename T>
__global__ void compute_dropout_forward_kernel(const T *input_data, T *output_data, T *mask_data,
                                               size_t batch_size, size_t channels,
                                               size_t spatial_size, T dropout_rate, T scale,
                                               unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  curandState_t state;
  curand_init(seed, idx, 0, &state);

  T rand_val = static_cast<T>(curand_uniform_double(&state));

  if (rand_val < dropout_rate) {
    mask_data[idx] = T(0);
    output_data[idx] = T(0);
  } else {
    mask_data[idx] = scale;
    output_data[idx] = input_data[idx] * scale;
  }
}

template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, T *mask_data, size_t batch_size,
                             size_t channels, size_t spatial_size, T dropout_rate,
                             cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

  T scale = T(1) / (T(1) - dropout_rate);
  unsigned long long seed = static_cast<unsigned long long>(std::time(nullptr));

  compute_dropout_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      input_data, output_data, mask_data, batch_size, channels, spatial_size, dropout_rate, scale,
      seed);

  cudaStreamSynchronize(stream);
}

template void compute_dropout_forward<float>(const float *input_data, float *output_data,
                                             float *mask_data, size_t batch_size, size_t channels,
                                             size_t spatial_size, float dropout_rate,
                                             cudaStream_t stream);
template void compute_dropout_forward<double>(const double *input_data, double *output_data,
                                              double *mask_data, size_t batch_size, size_t channels,
                                              size_t spatial_size, double dropout_rate,
                                              cudaStream_t stream);

} // namespace cuda
} // namespace tnn