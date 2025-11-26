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
namespace maxpool {
template <typename T>
void compute_max_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, size_t pad_h, size_t pad_w,
                              device_ptr<size_t[]> &mask_indices) {
  const T MIN_VALUE = std::numeric_limits<T>::lowest();

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const size_t input_offset = (n * channels + c) * input_h * input_w;
    const size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {

        // Calculate the theoretical window in the input (implicit padding)
        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);
        long h_end = h_start + pool_h;
        long w_end = w_start + pool_w;

        // Clamp the window to actual image boundaries
        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid = std::min(static_cast<long>(input_h), h_end);
        long w_end_valid = std::min(static_cast<long>(input_w), w_end);

        T max_val = MIN_VALUE;
        size_t max_idx = 0;

        // Loop only over valid pixels
        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {

            const size_t cur_input_idx = input_offset + ih * input_w + iw;
            T val = input_data[cur_input_idx];

            if (val > max_val) {
              max_val = val;
              max_idx = cur_input_idx;
            }
          }
        }

        const size_t out_idx = output_offset + out_h * output_w + out_w;
        output_data[out_idx] = max_val;
        mask_indices.get()[out_idx] = max_idx;
      }
    }
  });
}

template <typename T>
void compute_max_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const device_ptr<size_t[]> &mask_indices) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t i = 0; i < output_h * output_w; ++i) {
      const size_t out_idx = output_offset + i;
      const size_t input_idx = mask_indices.get()[out_idx];
      grad_input_data[input_idx] += gradient_data[out_idx];
    }
  });
}

// Explicit template instantiations
template void compute_max_pool_forward<float>(const float *input_data, float *output_data,
                                              size_t batch_size, size_t channels, size_t input_h,
                                              size_t input_w, size_t output_h, size_t output_w,
                                              size_t pool_h, size_t pool_w, size_t stride_h,
                                              size_t stride_w, size_t pad_h, size_t pad_w,
                                              device_ptr<size_t[]> &mask_indices);
template void compute_max_pool_forward<double>(const double *input_data, double *output_data,
                                               size_t batch_size, size_t channels, size_t input_h,
                                               size_t input_w, size_t output_h, size_t output_w,
                                               size_t pool_h, size_t pool_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w,
                                               device_ptr<size_t[]> &mask_indices);

template void compute_max_pool_backward<float>(const float *gradient_data, float *grad_input_data,
                                               size_t batch_size, size_t channels, size_t output_h,
                                               size_t output_w,
                                               const device_ptr<size_t[]> &mask_indices);
template void compute_max_pool_backward<double>(const double *gradient_data,
                                                double *grad_input_data, size_t batch_size,
                                                size_t channels, size_t output_h, size_t output_w,
                                                const device_ptr<size_t[]> &mask_indices);

} // namespace maxpool
} // namespace cpu
} // namespace tnn
