/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/maxpool_ops.hpp"

#include <limits>

#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {

template <typename T>
void compute_max_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, std::vector<size_t> &mask_indices) {
  const T MIN_VALUE = std::numeric_limits<T>::lowest();

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        T max_val = MIN_VALUE;
        size_t max_idx = 0;
        for (size_t ph = 0; ph < pool_h; ++ph) {
          for (size_t pw = 0; pw < pool_w; ++pw) {
            const size_t h_idx = out_h * stride_h + ph;
            const size_t w_idx = out_w * stride_w + pw;

            const size_t target_padded_idx =
                ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
            T val = input_data[target_padded_idx];
            if (val > max_val) {
              max_val = val;
              max_idx = target_padded_idx;
            }
          }
        }

        const size_t output_idx = ((n * channels + c) * output_h + out_h) * output_w + out_w;
        output_data[output_idx] = max_val;
        mask_indices[output_idx] = max_idx;
      }
    }
  });
}

template <typename T>
void compute_max_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const std::vector<size_t> &mask_indices) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        const size_t output_idx = ((n * channels + c) * output_h + out_h) * output_w + out_w;
        const T grad_val = gradient_data[output_idx];
        const size_t input_idx = mask_indices[output_idx];
        grad_input_data[input_idx] += grad_val;
      }
    }
  });
}

// Explicit template instantiations
template void compute_max_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w, std::vector<size_t> &mask_indices);
template void compute_max_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w, std::vector<size_t> &mask_indices);

template void compute_max_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t output_h,
                                               size_t output_w,
                                               const std::vector<size_t> &mask_indices);
template void compute_max_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t output_h, size_t output_w,
                                                const std::vector<size_t> &mask_indices);

} // namespace cpu
} // namespace tnn
