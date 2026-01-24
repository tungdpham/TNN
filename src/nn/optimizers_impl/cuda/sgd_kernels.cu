/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cuda/sgd_kernels.hpp"
#include "type/type.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {
namespace sgd {

template <typename T>
__global__ void update_sgd_kernel(T *params_data, const T *grads_data, const size_t size,
                                  const float learning_rate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    params_data[idx] -= learning_rate * static_cast<float>(grads_data[idx]);
  }
}

template <typename T>
__global__ void update_sgd_momentum_kernel(T *params_data, const T *grads_data, T *velocity_data,
                                           const size_t size, const float learning_rate,
                                           const float momentum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    velocity_data[idx] = momentum * static_cast<float>(velocity_data[idx]) -
                         learning_rate * static_cast<float>(grads_data[idx]);
    params_data[idx] += velocity_data[idx];
  }
}

template <typename T>
void update_sgd(T *params_data, const T *grads_data, const size_t size, const float learning_rate,
                cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  update_sgd_kernel<<<num_blocks, threads_per_block, 0, stream>>>(params_data, grads_data, size,
                                                                  learning_rate);
}

template <typename T>
void update_sgd_momentum(T *params_data, const T *grads_data, T *velocity_data, const size_t size,
                         const float learning_rate, const float momentum, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  update_sgd_momentum_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      params_data, grads_data, velocity_data, size, learning_rate, momentum);
}

#define INSTANTIATE_SGD_KERNELS(T)                                                                 \
  template void update_sgd<T>(T * params_data, const T *grads_data, const size_t size,             \
                              const float learning_rate, cudaStream_t stream);                     \
  template void update_sgd_momentum<T>(T * params_data, const T *grads_data, T *velocity_data,     \
                                       const size_t size, const float learning_rate,               \
                                       const float momentum, cudaStream_t stream);
INSTANTIATE_SGD_KERNELS(fp16)
INSTANTIATE_SGD_KERNELS(float)
INSTANTIATE_SGD_KERNELS(double)
#undef INSTANTIATE_SGD_KERNELS

} // namespace sgd
} // namespace cuda
} // namespace tnn

#endif
