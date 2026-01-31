#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T>
void relu(const T *input, T *output, size_t size);

template <typename T>
void relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size);

}  // namespace cpu
}  // namespace tnn