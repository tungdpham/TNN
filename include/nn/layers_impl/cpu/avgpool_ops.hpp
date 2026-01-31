/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T>
void avgpool_forward(const T *input, T *output, size_t batch_size, size_t height, size_t width,
                     size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                     size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);

template <typename T>
void avgpool_backward(const T *grad_output, T *grad_input, size_t batch_size, size_t input_h,
                      size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w);

}  // namespace cpu
}  // namespace tnn
