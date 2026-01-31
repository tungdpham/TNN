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
void softmax_forward(const T *input, T *output, size_t rows, size_t cols);

template <typename T>
void softmax_backward(const T *output, const T *grad_output, T *grad_input, size_t rows,
                      size_t cols);

}  // namespace cpu
}  // namespace tnn
