/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <vector>

namespace tnn {
namespace cpu {
namespace slice {

template <typename T>
void slice_forward(const T *input, T *output, const std::vector<size_t> &input_shape, size_t axis,
                   size_t start, size_t length);

template <typename T>
void slice_backward(const T *gradient, T *grad_input, const std::vector<size_t> &input_shape,
                    size_t axis, size_t start, size_t length);

} // namespace slice
} // namespace cpu
} // namespace tnn
