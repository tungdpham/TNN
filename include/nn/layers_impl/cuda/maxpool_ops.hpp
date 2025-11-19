#pragma once

#include "device/device_ptr.hpp"
#include <cstddef>

namespace tnn {
namespace cuda {
namespace maxpool {
template <typename T>
void compute_max_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, device_ptr<size_t[]> &mask_indices);

template <typename T>
void compute_max_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const device_ptr<size_t[]> &mask_indices);
} // namespace maxpool
} // namespace cuda
} // namespace tnn