/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/conv2d_ops.hpp"

#include "math/gemm.hpp"
#include "ops/cpu/kernels.hpp"
#include "threading/thread_handler.hpp"
#include "utils/mkl_utils.hpp"
#include <numeric>

namespace tnn {
namespace cpu {
namespace conv2d {
template <typename T>
void compute_conv_forward(const T *col_data, const T *weight_data, T *output_data,
                          const size_t output_size, const size_t kernel_size,
                          const size_t out_channels) {
#ifdef USE_MKL
  mkl::conv_forward_gemm(weight_data, col_data, output_data, static_cast<MKL_INT>(out_channels),
                         static_cast<MKL_INT>(kernel_size), static_cast<MKL_INT>(output_size));
#else
  gemm(weight_data, col_data, output_data, out_channels, output_size, kernel_size, false, false,
       T(1.0), T(0.0));
#endif
}

template <typename T>
void compute_weight_gradients(const T *col_data, const T *gradient_data, T *weight_grad_data,
                              const size_t output_size, const size_t kernel_size,
                              const size_t out_channels) {
#ifdef USE_MKL
  mkl::conv_weight_grad_gemm(gradient_data, col_data, weight_grad_data,
                             static_cast<MKL_INT>(out_channels), static_cast<MKL_INT>(kernel_size),
                             static_cast<MKL_INT>(output_size));
#else
  gemm(gradient_data, col_data, weight_grad_data, out_channels, kernel_size, output_size, false,
       true, T(1.0), T(1.0));
#endif
}

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *col_grad_data,
                             const size_t output_size, const size_t kernel_size,
                             const size_t out_channels) {
#ifdef USE_MKL
  mkl::conv_input_grad_gemm(weight_data, gradient_data, col_grad_data,
                            static_cast<MKL_INT>(out_channels), static_cast<MKL_INT>(kernel_size),
                            static_cast<MKL_INT>(output_size));
#else
  ops::cpu::set_scalar(col_grad_data, T(0), kernel_size * output_size);
  gemm(weight_data, gradient_data, col_grad_data, kernel_size, output_size, out_channels, true,
       false, T(1.0), T(1.0));
#endif
}

template <typename T>
void compute_bias_gradients(const T *gradient_data, T *bias_grad_data, const size_t batch_size,
                            const size_t output_h, const size_t output_w,
                            const size_t out_channels) {
  const size_t N_stride = out_channels * output_h * output_w;
  const size_t C_stride = output_h * output_w;

  parallel_for<size_t>(0, out_channels, [&](size_t oc) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      std::accumulate(gradient_data + n * N_stride + oc * C_stride,
                      gradient_data + n * N_stride + (oc + 1) * C_stride, grad_sum);
    }
    bias_grad_data[oc] += grad_sum;
  });
}

template <typename T>
void add_bias_to_output(T *output_data, const T *bias_data, const size_t batch_size,
                        const size_t output_h, const size_t output_w, const size_t out_channels) {
  parallel_for_2d(batch_size, out_channels, [&](size_t n, size_t oc) {
    ops::cpu::add_scalar(output_data + (n * out_channels + oc) * output_h * output_w, bias_data[oc],
                         output_data + (n * out_channels + oc) * output_h * output_w,
                         output_h * output_w);
  });
}

// Explicit template instantiations
template void compute_conv_forward<float>(const float *col_data, const float *weight_data,
                                          float *output_data, const size_t output_size,
                                          const size_t kernel_size, const size_t out_channels);

template void compute_weight_gradients<float>(const float *col_data, const float *gradient_data,
                                              float *weight_grad_data, const size_t output_size,
                                              const size_t kernel_size, const size_t out_channels);

template void compute_input_gradients<float>(const float *gradient_data, const float *weight_data,
                                             float *col_grad_data, const size_t output_size,
                                             const size_t kernel_size, const size_t out_channels);

template void compute_bias_gradients<float>(const float *gradient_data, float *bias_grad_data,
                                            const size_t batch_size, const size_t output_h,
                                            const size_t output_w, const size_t out_channels);

template void add_bias_to_output<float>(float *output_data, const float *bias_data,
                                        const size_t batch_size, const size_t output_h,
                                        const size_t output_w, const size_t out_channels);

} // namespace conv2d
} // namespace cpu
} // namespace tnn
