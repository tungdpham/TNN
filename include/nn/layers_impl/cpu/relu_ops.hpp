/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace tnn {
namespace cpu {
namespace relu {

template <typename T>
void relu_forward_with_mask(const T *input_data, T *output_data, uint8_t *mask_data,
                            size_t num_elements);

template <typename T>
void relu_backward_with_mask(const T *grad_output_data, T *grad_input_data,
                             const uint8_t *mask_data, size_t num_elements);

}  // namespace relu
}  // namespace cpu
}  // namespace tnn
