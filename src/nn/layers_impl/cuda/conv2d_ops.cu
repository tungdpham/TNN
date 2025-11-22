/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/conv2d_ops.hpp"

#include "cuda/error_handler.hpp"
#include "math/cuda/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace conv2d {
template <typename T>
__global__ void add_bias_kernel(T *output_data, const T *bias_data, size_t batch_size,
                                size_t output_h, size_t output_w, size_t out_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * out_channels * output_h * output_w;

  if (idx >= total_size)
    return;

  int remaining = idx % (out_channels * output_h * output_w);
  int c = remaining / (output_h * output_w);

  output_data[idx] += bias_data[c];
}

template <typename T>
__global__ void compute_bias_gradients_kernel(const T *gradient_data, T *bias_grad_data,
                                              size_t batch_size, size_t output_h, size_t output_w,
                                              size_t out_channels) {
  const size_t spatial_size = output_h * output_w;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = out_channels * spatial_size;

  int c = blockIdx.x;
  if (c >= out_channels)
    return;

  extern __shared__ char shared_mem[];
  T *shared_data = reinterpret_cast<T *>(shared_mem);

  T sum = T(0);

  int tid = threadIdx.x;
  int total_elements = batch_size * spatial_size;

  for (int idx = tid; idx < total_elements; idx += blockDim.x) {
    int n = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    sum += gradient_data[n * batch_stride + c * channel_stride + spatial_idx];
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
    bias_grad_data[c] = shared_data[0];
  }
}

template <typename T>
void compute_conv_forward(const T *col_data, const T *weight_data, T *output_data,
                          const size_t output_size, const size_t kernel_size,
                          const size_t out_channels, cudaStream_t stream) {
  cuda::gemm<T>(weight_data, col_data, output_data, out_channels, output_size, kernel_size, false,
                false, T(1.0), T(0.0), stream);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void compute_weight_gradients(const T *col_data, const T *gradient_data, T *weight_grad_data,
                              const size_t output_size, const size_t kernel_size,
                              const size_t out_channels, cudaStream_t stream) {
  cuda::gemm<T>(gradient_data, col_data, weight_grad_data, out_channels, kernel_size, output_size,
                false, true, T(1.0), T(1.0), stream);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *col_grad_data,
                             const size_t output_size, const size_t kernel_size,
                             const size_t out_channels, cudaStream_t stream) {
  cuda::gemm<T>(weight_data, gradient_data, col_grad_data, kernel_size, output_size, out_channels,
                true, false, T(1.0), T(0.0), stream);
}

template <typename T>
void compute_bias_gradients(const T *gradient_data, T *bias_grad_data, const size_t batch_size,
                            const size_t output_h, const size_t output_w, const size_t out_channels,
                            cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = out_channels;
  size_t shared_mem_size = threads_per_block * sizeof(T);

  compute_bias_gradients_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
      gradient_data, bias_grad_data, batch_size, output_h, output_w, out_channels);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void add_bias_to_output(T *output_data, const T *bias_data, const size_t batch_size,
                        const size_t output_h, const size_t output_w, const size_t out_channels,
                        cudaStream_t stream) {
  int total_size = batch_size * out_channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  add_bias_kernel<<<num_blocks, threads_per_block, 0, stream>>>(output_data, bias_data, batch_size,
                                                                output_h, output_w, out_channels);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template void compute_conv_forward<float>(const float *col_data, const float *weight_data,
                                          float *output_data, const size_t output_size,
                                          const size_t kernel_size, const size_t out_channels,
                                          cudaStream_t stream);

template void compute_weight_gradients<float>(const float *col_data, const float *gradient_data,
                                              float *weight_grad_data, const size_t output_size,
                                              const size_t kernel_size, const size_t out_channels,
                                              cudaStream_t stream);

template void compute_input_gradients<float>(const float *gradient_data, const float *weight_data,
                                             float *col_grad_data, const size_t output_size,
                                             const size_t kernel_size, const size_t out_channels,
                                             cudaStream_t stream);

template void compute_bias_gradients<float>(const float *gradient_data, float *bias_grad_data,
                                            const size_t batch_size, const size_t output_h,
                                            const size_t output_w, const size_t out_channels,
                                            cudaStream_t stream);

template void add_bias_to_output<float>(float *output_data, const float *bias_data,
                                        const size_t batch_size, const size_t output_h,
                                        const size_t output_w, const size_t out_channels,
                                        cudaStream_t stream);

} // namespace conv2d
} // namespace cuda
} // namespace tnn