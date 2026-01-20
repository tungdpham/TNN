/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/dense_ops.hpp"

#include "math/cuda/gemm.hpp"
#include "type/type.hpp"
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace dense {

template <typename T>
__global__ void add_bias_kernel(T *output_data, const T *bias_data, size_t batch_size,
                                size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size)
    return;

  int out_f = idx % output_features;
  output_data[idx] += bias_data[out_f];
}

template <typename T>
__global__ void compute_bias_gradients_kernel(const T *current_grad_data, T *bias_gradient_data,
                                              size_t batch_size, size_t output_features) {

  int out_f = blockIdx.x;
  if (out_f >= output_features)
    return;

  extern __shared__ char shared_mem[];
  T *shared_data = reinterpret_cast<T *>(shared_mem);

  T sum = T(0);

  int tid = threadIdx.x;
  for (int n = tid; n < batch_size; n += blockDim.x) {
    sum += current_grad_data[n * output_features + out_f];
  }

  shared_data[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    bias_gradient_data[out_f] += shared_data[0];
  }
}

template <typename T>
void compute_dense_forward(const T *input_data, const T *weight_data, T *output_data,
                           const size_t batch_size, const size_t input_features,
                           const size_t output_features, cudaStream_t stream) {

  tnn::cuda::gemm(input_data, weight_data, output_data, batch_size, output_features, input_features,
                  false, true, T(1.0f), T(0.0f), stream);
}

template <typename T>
void compute_weight_gradients(const T *input_data, const T *gradient_data, T *weight_grad_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features, cudaStream_t stream) {

  tnn::cuda::gemm(gradient_data, input_data, weight_grad_data, output_features, input_features,
                  batch_size, true, false, T(1.0f), T(1.0f), stream);
}

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *grad_input_data,
                             const size_t batch_size, const size_t input_features,
                             const size_t output_features, cudaStream_t stream) {
  tnn::cuda::gemm(gradient_data, weight_data, grad_input_data, batch_size, input_features,
                  output_features, false, false, T(1.0f), T(0.0f), stream);
}

template <typename T>
void compute_bias_gradients(const T *current_grad_data, const T *bias_gradient_data,
                            const size_t batch_size, const size_t output_features,
                            cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = output_features;
  size_t shared_mem_size = threads_per_block * sizeof(T);

  compute_bias_gradients_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
      current_grad_data, const_cast<T *>(bias_gradient_data), batch_size, output_features);
}

template <typename T>
void add_bias_vector(T *output_data, const T *bias_data, const size_t batch_size,
                     const size_t output_features, cudaStream_t stream) {
  int total_size = batch_size * output_features;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  add_bias_kernel<<<num_blocks, threads_per_block, 0, stream>>>(output_data, bias_data, batch_size,
                                                                output_features);
}
#define INSTANTIATE_DENSE(T)                                                                       \
  template void compute_dense_forward<T>(                                                          \
      const T *input_data, const T *weight_data, T *output_data, const size_t batch_size,          \
      const size_t input_features, const size_t output_features, cudaStream_t stream);             \
                                                                                                   \
  template void compute_weight_gradients<T>(                                                       \
      const T *input_data, const T *gradient_data, T *weight_grad_data, const size_t batch_size,   \
      const size_t input_features, const size_t output_features, cudaStream_t stream);             \
                                                                                                   \
  template void compute_input_gradients<T>(                                                        \
      const T *gradient_data, const T *weight_data, T *grad_input_data, const size_t batch_size,   \
      const size_t input_features, const size_t output_features, cudaStream_t stream);             \
                                                                                                   \
  template void compute_bias_gradients<T>(const T *current_grad_data, const T *bias_gradient_data, \
                                          const size_t batch_size, const size_t output_features,   \
                                          cudaStream_t stream);                                    \
                                                                                                   \
  template void add_bias_vector<T>(T * output_data, const T *bias_data, const size_t batch_size,   \
                                   const size_t output_features, cudaStream_t stream);
INSTANTIATE_DENSE(fp16)
INSTANTIATE_DENSE(float)
INSTANTIATE_DENSE(double)
#undef INSTANTIATE_DENSE
} // namespace dense
} // namespace cuda
} // namespace tnn
