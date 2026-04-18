/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cuda_runtime.h>

#include "math/cuda/gemm.hpp"
#include "nn/layers_impl/cuda/dense_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace legacy_dense {

template <typename IO_T, typename Param_T, typename Compute_T>
void run_forward(const IO_T* input_data, const Param_T* weight_data, IO_T* output_data,
                 const size_t batch_size, const size_t input_features, const size_t output_features,
                 cudaStream_t stream) {
  Compute_T alpha = static_cast<Compute_T>(1.0f);
  Compute_T beta = static_cast<Compute_T>(0.0f);

  cuda::gemm_ex<IO_T, Param_T, IO_T, Compute_T>(
      input_data, weight_data, output_data, batch_size, output_features, input_features, false,
      true, alpha, beta, input_features, input_features, output_features, stream);
}

template <typename IO_T, typename Param_T, typename Compute_T>
void run_wgrad(const IO_T* input_data, const IO_T* gradient_data, Param_T* weight_grad_data,
               const size_t batch_size, const size_t input_features, const size_t output_features,
               cudaStream_t stream) {
  Compute_T alpha = static_cast<Compute_T>(1.0f);
  Compute_T beta = static_cast<Compute_T>(1.0f);

  cuda::gemm_ex<IO_T, IO_T, Param_T, Compute_T>(
      gradient_data, input_data, weight_grad_data, output_features, input_features, batch_size,
      true, false, alpha, beta, output_features, input_features, input_features, stream);
}

template <typename IO_T, typename Param_T, typename Compute_T>
void run_dgrad(const IO_T* gradient_data, const Param_T* weight_data, IO_T* grad_input_data,
               const size_t batch_size, const size_t input_features, const size_t output_features,
               cudaStream_t stream) {
  Compute_T alpha = static_cast<Compute_T>(1.0f);
  Compute_T beta = static_cast<Compute_T>(0.0f);

  cuda::gemm_ex<IO_T, Param_T, IO_T, Compute_T>(
      gradient_data, weight_data, grad_input_data, batch_size, input_features, output_features,
      false, false, alpha, beta, output_features, input_features, input_features, stream);
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void run_bgrad_kernel_ex(const IO_T* current_grad_data, Param_T* bias_gradient_data,
                                    size_t batch_size, size_t output_features) {
  int out_f = blockIdx.x;
  if (out_f >= static_cast<int>(output_features)) return;

  extern __shared__ char shared_mem[];
  Compute_T* shared_data = reinterpret_cast<Compute_T*>(shared_mem);

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
void run_bgrad(const IO_T* current_grad_data, Param_T* bias_gradient_data, const size_t batch_size,
               const size_t output_features, cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = static_cast<int>(output_features);
  size_t shared_mem_size = threads_per_block * sizeof(Compute_T);

  run_bgrad_kernel_ex<IO_T, Param_T, Compute_T>
      <<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
          current_grad_data, bias_gradient_data, batch_size, output_features);
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void add_bias_kernel_ex(IO_T* output_data, const Param_T* bias_data, size_t batch_size,
                                   size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size) return;

  int out_f = idx % output_features;
  output_data[idx] += static_cast<IO_T>(bias_data[out_f]);
}

template <typename IO_T, typename Param_T, typename Compute_T>
void add_bias(IO_T* output_data, const Param_T* bias_data, const size_t batch_size,
              const size_t output_features, cudaStream_t stream) {
  int total_size = batch_size * output_features;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  add_bias_kernel_ex<IO_T, Param_T, Compute_T><<<num_blocks, threads_per_block, 0, stream>>>(
      output_data, bias_data, batch_size, output_features);
}

#define INSTANTIATE_3(IO_T, Param_T, Compute_T)                                            \
  template void run_forward<IO_T, Param_T, Compute_T>(                                     \
      const IO_T* input_data, const Param_T* weight_data, IO_T* output_data,               \
      const size_t batch_size, const size_t input_features, const size_t output_features,  \
      cudaStream_t stream);                                                                \
  template void run_wgrad<IO_T, Param_T, Compute_T>(                                       \
      const IO_T* input_data, const IO_T* gradient_data, Param_T* weight_grad_data,        \
      const size_t batch_size, const size_t input_features, const size_t output_features,  \
      cudaStream_t stream);                                                                \
  template void run_dgrad<IO_T, Param_T, Compute_T>(                                       \
      const IO_T* gradient_data, const Param_T* weight_data, IO_T* grad_input_data,        \
      const size_t batch_size, const size_t input_features, const size_t output_features,  \
      cudaStream_t stream);                                                                \
  template void run_bgrad<IO_T, Param_T, Compute_T>(                                       \
      const IO_T* current_grad_data, Param_T* bias_gradient_data, const size_t batch_size, \
      const size_t output_features, cudaStream_t stream);                                  \
  template void add_bias<IO_T, Param_T, Compute_T>(                                        \
      IO_T * output_data, const Param_T* bias_data, const size_t batch_size,               \
      const size_t output_features, cudaStream_t stream);

#define INSTANTIATE_2(IO_T, Param_T)   \
  INSTANTIATE_3(IO_T, Param_T, fp16)   \
  INSTANTIATE_3(IO_T, Param_T, bf16)   \
  INSTANTIATE_3(IO_T, Param_T, float)  \
  INSTANTIATE_3(IO_T, Param_T, double) \
  INSTANTIATE_3(IO_T, Param_T, int)

#define INSTANTIATE(IO_T)     \
  INSTANTIATE_2(IO_T, fp16)   \
  INSTANTIATE_2(IO_T, bf16)   \
  INSTANTIATE_2(IO_T, float)  \
  INSTANTIATE_2(IO_T, double) \
  INSTANTIATE_2(IO_T, int)

#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE
#undef INSTANTIATE_2
#undef INSTANTIATE_3

}  // namespace legacy_dense
}  // namespace cuda
}  // namespace tnn
