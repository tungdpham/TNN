/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/dense_ops.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace dense {

static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

template <typename T>
void cublas_gemm(const T *A, const T *B, T *C, int m, int n, int k, bool transA, bool transB);

template <>
void cublas_gemm<float>(const float *A, const float *B, float *C, int m, int n, int k, bool transA,
                        bool transB) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f, beta = 0.0f;

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(handle, opB, opA, n, m, k, &alpha, B, transB ? k : n, A, transA ? m : k, &beta, C, n);
}

template <>
void cublas_gemm<double>(const double *A, const double *B, double *C, int m, int n, int k,
                         bool transA, bool transB) {
  cublasHandle_t handle = get_cublas_handle();
  const double alpha = 1.0, beta = 0.0;

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasDgemm(handle, opB, opA, n, m, k, &alpha, B, transB ? k : n, A, transA ? m : k, &beta, C, n);
}

template <typename T> __global__ void set_zero_kernel(T *data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = T(0);
  }
}

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
  int out_f = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_f >= output_features)
    return;

  T grad_sum = T(0);
  for (size_t n = 0; n < batch_size; ++n) {
    grad_sum += current_grad_data[n * output_features + out_f];
  }

  atomicAdd(&bias_gradient_data[out_f], grad_sum);
}

template <typename T>
void compute_dense_forward(const T *input_data, const T *weight_data, T *output_data,
                           const size_t batch_size, const size_t input_features,
                           const size_t output_features) {

  cublas_gemm(input_data, weight_data, output_data, static_cast<int>(batch_size),
              static_cast<int>(output_features), static_cast<int>(input_features), false, true);
}

template <typename T>
void compute_weight_gradients(const T *input_data, const T *gradient_data, T *weight_grad_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features) {

  cublas_gemm(gradient_data, input_data, weight_grad_data, static_cast<int>(output_features),
              static_cast<int>(input_features), static_cast<int>(batch_size), true, false);
}

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *grad_input_data,
                             const size_t batch_size, const size_t input_features,
                             const size_t output_features) {

  int total_size = batch_size * input_features;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  set_zero_kernel<<<num_blocks, threads_per_block>>>(grad_input_data, total_size);

  cublas_gemm(gradient_data, weight_data, grad_input_data, static_cast<int>(batch_size),
              static_cast<int>(input_features), static_cast<int>(output_features), false, false);
}

template <typename T>
void compute_bias_gradients(const T *current_grad_data, const T *bias_gradient_data,
                            const size_t batch_size, const size_t output_features) {
  int threads_per_block = 256;
  int num_blocks = (output_features + threads_per_block - 1) / threads_per_block;

  compute_bias_gradients_kernel<<<num_blocks, threads_per_block>>>(
      current_grad_data, const_cast<T *>(bias_gradient_data), batch_size, output_features);
}

template <typename T>
void add_bias_vector(T *output_data, const T *bias_data, const size_t batch_size,
                     const size_t output_features) {
  int total_size = batch_size * output_features;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  add_bias_kernel<<<num_blocks, threads_per_block>>>(output_data, bias_data, batch_size,
                                                     output_features);
}

template void compute_dense_forward<float>(const float *input_data, const float *weight_data,
                                           float *output_data, const size_t batch_size,
                                           const size_t input_features,
                                           const size_t output_features);
template void compute_dense_forward<double>(const double *input_data, const double *weight_data,
                                            double *output_data, const size_t batch_size,
                                            const size_t input_features,
                                            const size_t output_features);

template void compute_weight_gradients<float>(const float *input_data, const float *gradient_data,
                                              float *weight_grad_data, const size_t batch_size,
                                              const size_t input_features,
                                              const size_t output_features);
template void compute_weight_gradients<double>(const double *input_data,
                                               const double *gradient_data,
                                               double *weight_grad_data, const size_t batch_size,
                                               const size_t input_features,
                                               const size_t output_features);

template void compute_input_gradients<float>(const float *gradient_data, const float *weight_data,
                                             float *grad_input_data, const size_t batch_size,
                                             const size_t input_features,
                                             const size_t output_features);
template void compute_input_gradients<double>(const double *gradient_data,
                                              const double *weight_data, double *grad_input_data,
                                              const size_t batch_size, const size_t input_features,
                                              const size_t output_features);

template void compute_bias_gradients<float>(const float *current_grad_data,
                                            const float *bias_gradient_data,
                                            const size_t batch_size, const size_t output_features);
template void compute_bias_gradients<double>(const double *current_grad_data,
                                             const double *bias_gradient_data,
                                             const size_t batch_size, const size_t output_features);

template void add_bias_vector<float>(float *output_data, const float *bias_data,
                                     const size_t batch_size, const size_t output_features);
template void add_bias_vector<double>(double *output_data, const double *bias_data,
                                      const size_t batch_size, const size_t output_features);
} // namespace dense
} // namespace cuda
} // namespace tnn
