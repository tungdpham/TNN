#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T> void elu(const T *input, T *output, size_t size, T alpha);

template <typename T> void elu_gradient(const T *input, T *grad_output, size_t size, T alpha);

} // namespace cpu
} // namespace tnn
