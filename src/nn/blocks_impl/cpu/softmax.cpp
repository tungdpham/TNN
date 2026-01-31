/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/cpu/softmax.hpp"

#include <algorithm>
#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {

template <typename T>
void softmax_forward(const T *input, T *output, size_t rows, size_t cols) {
  parallel_for<size_t>(0, rows, [&](size_t i) {
    const T *row_in = input + i * cols;
    T *row_out = output + i * cols;

    T max_val = -INFINITY;
    for (size_t j = 0; j < cols; ++j) {
      if (row_in[j] > max_val) max_val = row_in[j];
    }

    if (std::isinf(static_cast<float>(max_val)) && max_val < T(0)) max_val = T(0);

    T sum = 0;
    for (size_t j = 0; j < cols; ++j) {
      T val = std::exp(static_cast<float>(row_in[j]) - static_cast<float>(max_val));
      row_out[j] = val;
      sum += val;
    }

    T inv_sum = T(1.0) / std::max(sum, static_cast<T>(1e-8));
    for (size_t j = 0; j < cols; ++j) {
      row_out[j] *= inv_sum;
    }
  });
}

template <typename T>
void softmax_backward(const T *output, const T *grad_output, T *grad_input, size_t rows,
                      size_t cols) {
  // dX_j = Y_j * (dY_j - sum(dY_k * Y_k))
  parallel_for<size_t>(0, rows, [&](size_t i) {
    const T *y = output + i * cols;
    const T *dy = grad_output + i * cols;
    T *dx = grad_input + i * cols;

    T sum_dy_y = 0;
    for (size_t j = 0; j < cols; ++j) {
      sum_dy_y += dy[j] * y[j];
    }

    for (size_t j = 0; j < cols; ++j) {
      dx[j] = y[j] * (dy[j] - sum_dy_y);
    }
  });
}

#define INSTANTIATE_SOFTMAX(T)                                                            \
  template void softmax_forward<T>(const T *input, T *output, size_t rows, size_t cols);  \
                                                                                          \
  template void softmax_backward<T>(const T *output, const T *grad_output, T *grad_input, \
                                    size_t rows, size_t cols);

INSTANTIATE_SOFTMAX(fp16)
INSTANTIATE_SOFTMAX(bf16)
INSTANTIATE_SOFTMAX(float)
INSTANTIATE_SOFTMAX(double)
#undef INSTANTIATE_SOFTMAX

}  // namespace cpu
}  // namespace tnn
