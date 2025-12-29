#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T> void elu(const T *input, T *output, size_t size, T alpha);

template <typename T>
void elu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size, T alpha);

} // namespace cpu
} // namespace tnn
