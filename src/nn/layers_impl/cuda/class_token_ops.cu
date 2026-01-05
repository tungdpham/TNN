/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/class_token_ops.hpp"
#include <cstdio>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {

template <typename T>
__global__ void class_token_forward_imp(const T *input, const T *token, T *output, size_t channels,
                                        size_t input_spatial, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  size_t output_spatial = input_spatial + 1;
  size_t s_out = idx % output_spatial;
  size_t tmp = idx / output_spatial;
  size_t c = tmp % channels;
  size_t n = tmp / channels;

  if (s_out == 0) {
    output[idx] = token[c];
  } else {
    size_t s_in = s_out - 1;
    size_t in_idx = n * channels * input_spatial + c * input_spatial + s_in;
    output[idx] = input[in_idx];
  }
}

template <typename T>
void class_token_forward(const T *input, const T *token, T *output, size_t batch_size,
                         size_t channels, size_t spatial_size, cudaStream_t stream) {
  size_t output_spatial = spatial_size + 1;
  size_t total_elements = batch_size * channels * output_spatial;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  class_token_forward_imp<T><<<blocks, threads, 0, stream>>>(input, token, output, channels,
                                                             spatial_size, total_elements);
}

template <typename T>
__global__ void class_token_backward_input_imp(const T *grad_output, T *grad_input, size_t channels,
                                               size_t input_spatial, size_t total_input_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_input_elements)
    return;

  size_t s_in = idx % input_spatial;
  size_t tmp = idx / input_spatial;
  size_t c = tmp % channels;
  size_t n = tmp / channels;

  size_t output_spatial = input_spatial + 1;
  size_t out_idx = n * channels * output_spatial + c * output_spatial + (s_in + 1);

  grad_input[idx] = grad_output[out_idx];
}

template <typename T>
__global__ void class_token_backward_token_imp(const T *grad_output, T *grad_token,
                                               size_t batch_size, size_t channels,
                                               size_t input_spatial) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_items = batch_size * channels;
  if (idx >= total_items)
    return;

  size_t c = idx % channels;
  size_t n = idx / channels;

  size_t output_spatial = input_spatial + 1;
  size_t out_idx = n * channels * output_spatial + c * output_spatial + 0;

  atomicAdd(&grad_token[c], grad_output[out_idx]);
}

template <typename T>
void class_token_backward(const T *grad_output, T *grad_input, T *grad_token, size_t batch_size,
                          size_t channels, size_t spatial_size, cudaStream_t stream) {
  size_t total_input = batch_size * channels * spatial_size;
  int threads = 256;
  int blocks = (total_input + threads - 1) / threads;
  class_token_backward_input_imp<T><<<blocks, threads, 0, stream>>>(
      grad_output, grad_input, channels, spatial_size, total_input);

  size_t total_token_contribs = batch_size * channels;
  blocks = (total_token_contribs + threads - 1) / threads;
  class_token_backward_token_imp<T>
      <<<blocks, threads, 0, stream>>>(grad_output, grad_token, batch_size, channels, spatial_size);
}

template void class_token_forward<float>(const float *input, const float *token, float *output,
                                         size_t batch_size, size_t channels, size_t spatial_size,
                                         cudaStream_t stream);
template void class_token_backward<float>(const float *grad_output, float *grad_input,
                                          float *grad_token, size_t batch_size, size_t channels,
                                          size_t spatial_size, cudaStream_t stream);

template void class_token_forward<double>(const double *input, const double *token, double *output,
                                          size_t batch_size, size_t channels, size_t spatial_size,
                                          cudaStream_t stream);
template void class_token_backward<double>(const double *grad_output, double *grad_input,
                                           double *grad_token, size_t batch_size, size_t channels,
                                           size_t spatial_size, cudaStream_t stream);

} // namespace cuda
} // namespace tnn
