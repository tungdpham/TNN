/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/cpu/relu_ops.hpp"

#include <cstddef>
#include <cstdint>

#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace relu {

template <typename T>
void relu_forward_with_mask(const T *input_data, T *output_data, uint8_t *mask_data,
                            size_t num_elements) {
  T zero = static_cast<T>(0);

  for (size_t i = 0; i < num_elements; ++i) {
    bool is_positive = input_data[i] > zero;
    output_data[i] = is_positive ? input_data[i] : zero;
    mask_data[i] = is_positive ? 1 : 0;
  }
}

template <typename T>
void relu_backward_with_mask(const T *grad_output_data, T *grad_input_data,
                             const uint8_t *mask_data, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    grad_input_data[i] = grad_output_data[i] * static_cast<T>(mask_data[i]);
  }
}

#define INSTANTIATE(T)                                                                             \
  template void relu_forward_with_mask<T>(const T *input_data, T *output_data, uint8_t *mask_data, \
                                          size_t num_elements);                                    \
                                                                                                   \
  template void relu_backward_with_mask<T>(const T *grad_output_data, T *grad_input_data,          \
                                           const uint8_t *mask_data, size_t num_elements);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace relu
}  // namespace cpu
}  // namespace tnn
