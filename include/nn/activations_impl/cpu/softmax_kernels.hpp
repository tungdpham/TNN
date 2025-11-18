#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T>
void softmax(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
             size_t width);

template <typename T>
void softmax_gradient(const T *input, T *grad_output, size_t batch_size, size_t channels,
                      size_t height, size_t width);

} // namespace cpu
} // namespace tnn
