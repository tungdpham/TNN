/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "cuda/error_handler.hpp"
#include "nn/layers_impl/cuda/avgpool_ops.hpp"
#include "type/type.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {

// Forward kernel for NHWC average pooling
template <typename T>
__global__ void avgpool_forward_kernel(const T *input, T *output, size_t batch_size, size_t height,
                                       size_t width, size_t channels, size_t pool_h, size_t pool_w,
                                       size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                       size_t output_h, size_t output_w) {
  // Calculate output position
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs)
    return;

  // Decode NHWC indices
  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  // Calculate input window bounds
  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  // Compute average
  float sum = 0.0f;
  int count = 0;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      sum += static_cast<float>(input[input_idx]);
      ++count;
    }
  }

  output[idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
}

// Backward kernel for NHWC average pooling
template <typename T>
__global__ void avgpool_backward_kernel(const T *grad_output, T *grad_input, size_t batch_size,
                                        size_t input_h, size_t input_w, size_t channels,
                                        size_t pool_h, size_t pool_w, size_t stride_h,
                                        size_t stride_w, size_t pad_h, size_t pad_w,
                                        size_t output_h, size_t output_w) {
  // Calculate output position
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs)
    return;

  // Decode NHWC indices
  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  // Get gradient value
  float grad = static_cast<float>(grad_output[idx]);

  // Calculate input window bounds
  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  // Count valid elements
  int count = (h_end - h_start) * (w_end - w_start);
  if (count == 0)
    return;

  // Distribute gradient evenly
  float grad_per_element = grad / count;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
      atomicAdd(&grad_input[input_idx], static_cast<T>(grad_per_element));
    }
  }
}

// Specialization for half precision atomicAdd
template <>
__global__ void avgpool_backward_kernel<half>(const half *grad_output, half *grad_input,
                                              size_t batch_size, size_t input_h, size_t input_w,
                                              size_t channels, size_t pool_h, size_t pool_w,
                                              size_t stride_h, size_t stride_w, size_t pad_h,
                                              size_t pad_w, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs)
    return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  float grad = __half2float(grad_output[idx]);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  int count = (h_end - h_start) * (w_end - w_start);
  if (count == 0)
    return;

  float grad_per_element = grad / count;
  __half grad_half = __float2half(grad_per_element);

  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
      atomicAdd(reinterpret_cast<__half *>(&grad_input[input_idx]), grad_half);
    }
  }
}

template <typename T>
void avgpool_forward(const T *input, T *output, size_t batch_size, size_t height, size_t width,
                     size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                     size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                     size_t output_w) {
  size_t total_outputs = batch_size * output_h * output_w * channels;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;

  avgpool_forward_kernel<<<blocks, threads>>>(input, output, batch_size, height, width, channels,
                                              pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                              output_h, output_w);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void avgpool_backward(const T *grad_output, T *grad_input, size_t batch_size, size_t input_h,
                      size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {

  size_t total_outputs = batch_size * output_h * output_w * channels;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;

  avgpool_backward_kernel<<<blocks, threads>>>(grad_output, grad_input, batch_size, input_h,
                                               input_w, channels, pool_h, pool_w, stride_h,
                                               stride_w, pad_h, pad_w, output_h, output_w);

  CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_AVGPOOL_FUNCS(T)                                                               \
  template void avgpool_forward<T>(const T *input, T *output, size_t batch_size, size_t height,    \
                                   size_t width, size_t channels, size_t pool_h, size_t pool_w,    \
                                   size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,   \
                                   size_t output_h, size_t output_w);                              \
  template void avgpool_backward<T>(                                                               \
      const T *grad_output, T *grad_input, size_t batch_size, size_t input_h, size_t input_w,      \
      size_t channels, size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,             \
      size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);

INSTANTIATE_AVGPOOL_FUNCS(fp16)
INSTANTIATE_AVGPOOL_FUNCS(float)
INSTANTIATE_AVGPOOL_FUNCS(double)
#undef INSTANTIATE_AVGPOOL_FUNCS

} // namespace cuda
} // namespace tnn
