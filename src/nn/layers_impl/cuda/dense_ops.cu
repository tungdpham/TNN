/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "math/cuda/gemm.hpp"
#include "nn/layers_impl/cuda/dense_ops.hpp"
#include "type/type.hpp"
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace dense {

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_dense_forward_ex(const IO_T *input_data, const Param_T *weight_data, IO_T *output_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features, cudaStream_t stream) {
  Compute_T alpha = static_cast<Compute_T>(1.0f);
  Compute_T beta = static_cast<Compute_T>(0.0f);

  cuda::gemm_ex<IO_T, Param_T, IO_T, Compute_T>(
      input_data, weight_data, output_data, batch_size, output_features, input_features, false,
      true, alpha, beta, input_features, input_features, output_features, stream);
}

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_weight_gradients_ex(const IO_T *input_data, const IO_T *gradient_data,
                                 Param_T *weight_grad_data, const size_t batch_size,
                                 const size_t input_features, const size_t output_features,
                                 cudaStream_t stream) {
  Compute_T alpha = static_cast<Compute_T>(1.0f);
  Compute_T beta = static_cast<Compute_T>(1.0f);

  cuda::gemm_ex<IO_T, IO_T, Param_T, Compute_T>(
      gradient_data, input_data, weight_grad_data, output_features, input_features, batch_size,
      true, false, alpha, beta, output_features, input_features, input_features, stream);
}

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_input_gradients_ex(const IO_T *gradient_data, const Param_T *weight_data,
                                IO_T *grad_input_data, const size_t batch_size,
                                const size_t input_features, const size_t output_features,
                                cudaStream_t stream) {
  Compute_T alpha = static_cast<Compute_T>(1.0f);
  Compute_T beta = static_cast<Compute_T>(0.0f);

  cuda::gemm_ex<IO_T, Param_T, IO_T, Compute_T>(
      gradient_data, weight_data, grad_input_data, batch_size, input_features, output_features,
      false, false, alpha, beta, output_features, input_features, input_features, stream);
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void compute_bias_gradients_kernel_ex(const IO_T *current_grad_data,
                                                 Param_T *bias_gradient_data, size_t batch_size,
                                                 size_t output_features) {
  int out_f = blockIdx.x;
  if (out_f >= static_cast<int>(output_features))
    return;

  extern __shared__ char shared_mem[];
  Compute_T *shared_data = reinterpret_cast<Compute_T *>(shared_mem);

  Compute_T sum = Compute_T(0);
  int tid = threadIdx.x;
  for (int n = tid; n < static_cast<int>(batch_size); n += blockDim.x) {
    sum += static_cast<Compute_T>(current_grad_data[n * output_features + out_f]);
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
    Compute_T prev = static_cast<Compute_T>(bias_gradient_data[out_f]);
    bias_gradient_data[out_f] = static_cast<Param_T>(prev + shared_data[0]);
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_bias_gradients_ex(const IO_T *current_grad_data, Param_T *bias_gradient_data,
                               const size_t batch_size, const size_t output_features,
                               cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = static_cast<int>(output_features);
  size_t shared_mem_size = threads_per_block * sizeof(Compute_T);

  compute_bias_gradients_kernel_ex<IO_T, Param_T, Compute_T>
      <<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
          current_grad_data, bias_gradient_data, batch_size, output_features);
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void add_bias_kernel_ex(IO_T *output_data, const Param_T *bias_data, size_t batch_size,
                                   size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size)
    return;

  int out_f = idx % output_features;
  output_data[idx] += static_cast<IO_T>(bias_data[out_f]);
}

template <typename IO_T, typename Param_T, typename Compute_T>
void add_bias_vector_ex(IO_T *output_data, const Param_T *bias_data, const size_t batch_size,
                        const size_t output_features, cudaStream_t stream) {
  int total_size = batch_size * output_features;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  add_bias_kernel_ex<IO_T, Param_T, Compute_T><<<num_blocks, threads_per_block, 0, stream>>>(
      output_data, bias_data, batch_size, output_features);
}

#define INSTANTIATE_DENSE_OPS(IO_T, Param_T, Compute_T)                                            \
  template void compute_dense_forward_ex<IO_T, Param_T, Compute_T>(                                \
      const IO_T *input_data, const Param_T *weight_data, IO_T *output_data,                       \
      const size_t batch_size, const size_t input_features, const size_t output_features,          \
      cudaStream_t stream);                                                                        \
  template void compute_weight_gradients_ex<IO_T, Param_T, Compute_T>(                             \
      const IO_T *input_data, const IO_T *gradient_data, Param_T *weight_grad_data,                \
      const size_t batch_size, const size_t input_features, const size_t output_features,          \
      cudaStream_t stream);                                                                        \
  template void compute_input_gradients_ex<IO_T, Param_T, Compute_T>(                              \
      const IO_T *gradient_data, const Param_T *weight_data, IO_T *grad_input_data,                \
      const size_t batch_size, const size_t input_features, const size_t output_features,          \
      cudaStream_t stream);                                                                        \
  template void compute_bias_gradients_ex<IO_T, Param_T, Compute_T>(                               \
      const IO_T *current_grad_data, Param_T *bias_gradient_data, const size_t batch_size,         \
      const size_t output_features, cudaStream_t stream);                                          \
  template void add_bias_vector_ex<IO_T, Param_T, Compute_T>(                                      \
      IO_T * output_data, const Param_T *bias_data, const size_t batch_size,                       \
      const size_t output_features, cudaStream_t stream);

#define INSTANTIATE_DENSE_OPS_COMPUTE(IO_T, Param_T, Compute_T)                                    \
  INSTANTIATE_DENSE_OPS(IO_T, Param_T, Compute_T)

#define INSTANTIATE_DENSE_OPS_PARAM(IO_T, Param_T)                                                 \
  INSTANTIATE_DENSE_OPS_COMPUTE(IO_T, Param_T, fp16)                                               \
  INSTANTIATE_DENSE_OPS_COMPUTE(IO_T, Param_T, bf16)                                               \
  INSTANTIATE_DENSE_OPS_COMPUTE(IO_T, Param_T, float)                                              \
  INSTANTIATE_DENSE_OPS_COMPUTE(IO_T, Param_T, double)

#define INSTANTIATE_DENSE_OPS_IO(IO_T)                                                             \
  INSTANTIATE_DENSE_OPS_PARAM(IO_T, fp16)                                                          \
  INSTANTIATE_DENSE_OPS_PARAM(IO_T, bf16)                                                          \
  INSTANTIATE_DENSE_OPS_PARAM(IO_T, float)                                                         \
  INSTANTIATE_DENSE_OPS_PARAM(IO_T, double)

INSTANTIATE_DENSE_OPS_IO(fp16)
INSTANTIATE_DENSE_OPS_IO(bf16)
INSTANTIATE_DENSE_OPS_IO(float)
INSTANTIATE_DENSE_OPS_IO(double)

#undef INSTANTIATE_DENSE_OPS_IO
#undef INSTANTIATE_DENSE_OPS_PARAM
#undef INSTANTIATE_DENSE_OPS_COMPUTE
#undef INSTANTIATE_DENSE_OPS

} // namespace dense
} // namespace cuda
} // namespace tnn
