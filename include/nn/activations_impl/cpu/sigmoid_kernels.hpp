#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T> void sigmoid(const T *input, T *output, size_t size);

template <typename T> void sigmoid_gradient(const T *input, T *grad_output, size_t size);

} // namespace cpu
} // namespace tnn
