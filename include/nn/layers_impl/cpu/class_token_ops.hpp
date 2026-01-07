#pragma once
#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T>
void class_token_forward(const T *input, const T *token, T *output, size_t batch_size,
                         size_t channels, size_t spatial_size);

template <typename T>
void class_token_backward(const T *grad_output, T *grad_input, T *grad_token, size_t batch_size,
                          size_t channels, size_t spatial_size);

} // namespace cpu
} // namespace tnn
