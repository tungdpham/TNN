/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/conv2d_nchw_ops.hpp"

#include <numeric>

#include "math/cpu/gemm.hpp"
#include "ops/cpu/kernels.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace conv2d_nchw {
template <typename T>
void run_forward(const T *col_data, const T *weight_data, T *output_data, const size_t output_size,
                 const size_t kernel_size, const size_t out_channels) {
  gemm(weight_data, col_data, output_data, out_channels, output_size, kernel_size, false, false,
       T(1.0), T(0.0));
}

template <typename T>
void run_wgrad(const T *col_data, const T *gradient_data, T *weight_grad_data,
               const size_t output_size, const size_t kernel_size, const size_t out_channels) {
  gemm(gradient_data, col_data, weight_grad_data, out_channels, kernel_size, output_size, false,
       true, T(1.0), T(1.0));
}

template <typename T>
void run_dgrad(const T *gradient_data, const T *weight_data, T *col_grad_data,
               const size_t output_size, const size_t kernel_size, const size_t out_channels) {
  gemm(weight_data, gradient_data, col_grad_data, kernel_size, output_size, out_channels, true,
       false, T(1.0), T(0.0));
}

template <typename T>
void run_bgrad(const T *gradient_data, T *bias_grad_data, const size_t batch_size,
               const size_t output_h, const size_t output_w, const size_t out_channels) {
  const size_t N_stride = out_channels * output_h * output_w;
  const size_t C_stride = output_h * output_w;

  parallel_for<size_t>(0, out_channels, [&](size_t oc) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      grad_sum = std::accumulate(gradient_data + n * N_stride + oc * C_stride,
                                 gradient_data + n * N_stride + (oc + 1) * C_stride, grad_sum);
    }
    bias_grad_data[oc] += grad_sum;
  });
}

template <typename T>
void add_bias(T *output_data, const T *bias_data, const size_t batch_size, const size_t output_h,
              const size_t output_w, const size_t out_channels) {
  parallel_for_2d(batch_size, out_channels, [&](size_t n, size_t oc) {
    ops::cpu::add_scalar(output_data + (n * out_channels + oc) * output_h * output_w, bias_data[oc],
                         output_data + (n * out_channels + oc) * output_h * output_w,
                         output_h * output_w);
  });
}

#define INSTANTIATE(T)                                                                           \
  template void run_forward<T>(const T *col_data, const T *weight_data, T *output_data,          \
                               const size_t output_size, const size_t kernel_size,               \
                               const size_t out_channels);                                       \
                                                                                                 \
  template void run_wgrad<T>(const T *col_data, const T *gradient_data, T *weight_grad_data,     \
                             const size_t output_size, const size_t kernel_size,                 \
                             const size_t out_channels);                                         \
                                                                                                 \
  template void run_dgrad<T>(const T *gradient_data, const T *weight_data, T *col_grad_data,     \
                             const size_t output_size, const size_t kernel_size,                 \
                             const size_t out_channels);                                         \
                                                                                                 \
  template void run_bgrad<T>(const T *gradient_data, T *bias_grad_data, const size_t batch_size, \
                             const size_t output_h, const size_t output_w,                       \
                             const size_t out_channels);                                         \
                                                                                                 \
  template void add_bias<T>(T * output_data, const T *bias_data, const size_t batch_size,        \
                            const size_t output_h, const size_t output_w,                        \
                            const size_t out_channels);

#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace conv2d_nchw
}  // namespace cpu
}  // namespace tnn
