/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda/error_handler.hpp"
#include "nn/layers_impl/cuda/maxpool_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {

// Forward kernel for NHWC max pooling
template <typename T>
__global__ void maxpool_forward_kernel(const T* input, T* output, int* mask_indices,
                                       size_t batch_size, size_t height, size_t width,
                                       size_t channels, size_t pool_h, size_t pool_w,
                                       size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                       size_t output_h, size_t output_w) {
  // Calculate output position
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

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

  // Find maximum value
  float max_val = -INFINITY;
  int max_idx = -1;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      float val = static_cast<float>(input[input_idx]);
      if (val > max_val) {
        max_val = val;
        max_idx = static_cast<int>(input_idx);
      }
    }
  }

  output[idx] = static_cast<T>(max_val);
  mask_indices[idx] = max_idx;
}

// Backward kernel for NHWC max pooling
template <typename T>
__global__ void maxpool_backward_kernel(const T* grad_output, T* grad_input,
                                        const int* mask_indices, size_t batch_size, size_t channels,
                                        size_t output_h, size_t output_w) {
  // Calculate output position
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  int max_idx = mask_indices[idx];
  if (max_idx >= 0) {
    atomicAdd(&grad_input[max_idx], grad_output[idx]);
  }
}

// Specialization for half precision atomicAdd
template <>
__global__ void maxpool_backward_kernel<half>(const half* grad_output, half* grad_input,
                                              const int* mask_indices, size_t batch_size,
                                              size_t channels, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  int max_idx = mask_indices[idx];
  if (max_idx >= 0) {
    atomicAdd(reinterpret_cast<__half*>(&grad_input[max_idx]),
              *reinterpret_cast<const __half*>(&grad_output[idx]));
  }
}

template <typename T>
void maxpool_forward(const T* input, T* output, int* mask_indices, size_t batch_size, size_t height,
                     size_t width, size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                     size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                     size_t output_w) {
  size_t total_outputs = batch_size * output_h * output_w * channels;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;

  maxpool_forward_kernel<<<blocks, threads>>>(input, output, mask_indices, batch_size, height,
                                              width, channels, pool_h, pool_w, stride_h, stride_w,
                                              pad_h, pad_w, output_h, output_w);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void maxpool_backward(const T* grad_output, T* grad_input, const int* mask_indices,
                      size_t batch_size, size_t channels, size_t output_h, size_t output_w) {
  size_t total_outputs = batch_size * output_h * output_w * channels;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;

  maxpool_backward_kernel<<<blocks, threads>>>(grad_output, grad_input, mask_indices, batch_size,
                                               channels, output_h, output_w);

  CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_MAXPOOL_FUNCS(T)                                                              \
  template void maxpool_forward<T>(                                                               \
      const T* input, T* output, int* mask_indices, size_t batch_size, size_t height,             \
      size_t width, size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,               \
      size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);             \
  template void maxpool_backward<T>(const T* grad_output, T* grad_input, const int* mask_indices, \
                                    size_t batch_size, size_t channels, size_t output_h,          \
                                    size_t output_w);
INSTANTIATE_MAXPOOL_FUNCS(fp16)
INSTANTIATE_MAXPOOL_FUNCS(bf16)
INSTANTIATE_MAXPOOL_FUNCS(float)
INSTANTIATE_MAXPOOL_FUNCS(double)
#undef INSTANTIATE_MAXPOOL_FUNCS

}  // namespace cuda
}  // namespace tnn
