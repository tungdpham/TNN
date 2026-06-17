#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>

namespace synet {
namespace cuda {
namespace conv2d_nchw {
template <typename T>
void run_forward(const T *col_data, const T *weight_data, T *output_data, size_t output_size,
                 size_t kernel_size, size_t out_channels, cudaStream_t stream);
template <typename T>
void run_wgrad(const T *col_data, const T *gradient_data, T *weight_grad_data, size_t output_size,
               size_t kernel_size, size_t out_channels, cudaStream_t stream);

template <typename T>
void run_dgrad(const T *gradient_data, const T *weight_data, T *col_grad_data, size_t output_size,
               size_t kernel_size, size_t out_channels, cudaStream_t stream);

template <typename T>
void run_bgrad(const T *gradient_data, T *bias_grad_data, size_t batch_size, size_t output_h,
               size_t output_w, size_t out_channels, cudaStream_t stream);

template <typename T>
void add_bias(T *output_data, const T *bias_data, size_t batch_size, size_t output_h,
              size_t output_w, size_t out_channels, cudaStream_t stream);
}  // namespace conv2d_nchw
}  // namespace cuda
}  // namespace synet
#endif