#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T> void leaky_relu(const T *input, T *output, size_t size, T negative_slope);

template <typename T>
void leaky_relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size,
                         T negative_slope);

} // namespace cpu
} // namespace tnn
