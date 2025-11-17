/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/avgpool_ops.hpp"

#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {
namespace avgpool {
template <typename T>
void compute_avg_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        T sum = T(0);
        for (size_t ph = 0; ph < pool_h; ++ph) {
          for (size_t pw = 0; pw < pool_w; ++pw) {
            const size_t h_idx = out_h * stride_h + ph;
            const size_t w_idx = out_w * stride_w + pw;

            const size_t input_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
            sum += input_data[input_idx];
          }
        }

        const size_t output_idx = ((n * channels + c) * output_h + out_h) * output_w + out_w;
        output_data[output_idx] = sum * pool_size_inv;
      }
    }
  });
}

template <typename T>
void compute_avg_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        const size_t output_idx = ((n * channels + c) * output_h + out_h) * output_w + out_w;
        const T grad_val = gradient_data[output_idx] * pool_size_inv;

        for (size_t ph = 0; ph < pool_h; ++ph) {
          for (size_t pw = 0; pw < pool_w; ++pw) {
            const size_t h_idx = out_h * stride_h + ph;
            const size_t w_idx = out_w * stride_w + pw;

            const size_t input_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
            grad_input_data[input_idx] += grad_val;
          }
        }
      }
    }
  });
}

// Explicit template instantiations
template void compute_avg_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w);
template void compute_avg_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w);

template void compute_avg_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w);
template void compute_avg_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t input_h, size_t input_w,
                                                size_t output_h, size_t output_w, size_t pool_h,
                                                size_t pool_w, size_t stride_h, size_t stride_w);
} // namespace avgpool
} // namespace cpu
} // namespace tnn
