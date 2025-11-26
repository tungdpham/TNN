#pragma once

#ifdef USE_CUDA
#include "device/device_ptr.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace maxpool {
template <typename T>
void compute_max_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, size_t pad_h, size_t pad_w,
                              device_ptr<size_t[]> &mask_indices, cudaStream_t stream);

template <typename T>
void compute_max_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const device_ptr<size_t[]> &mask_indices, cudaStream_t stream);
} // namespace maxpool
} // namespace cuda
} // namespace tnn

#endif