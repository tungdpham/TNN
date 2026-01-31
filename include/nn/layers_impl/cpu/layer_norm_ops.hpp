#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace layer_norm {

template <typename T>
void layer_norm_forward(const T *input, T *output, const T *gamma, const T *beta, size_t batch_size,
                        size_t channels, T epsilon);

template <typename T>
void layer_norm_backward(const T *grad_output, const T *input, const T *gamma, T *grad_input,
                         T *grad_gamma, T *grad_beta, size_t batch_size, size_t channels,
                         T epsilon);

}  // namespace layer_norm
}  // namespace cpu
}  // namespace tnn
