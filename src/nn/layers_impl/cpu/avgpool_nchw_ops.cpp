/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/avgpool_nchw_ops.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace avgpool_nchw {
template <typename T>
void compute_avg_pool_forward(const T *input_data, T *output_data, size_t batch_size,
                              size_t channels, size_t input_h, size_t input_w, size_t output_h,
                              size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                              size_t stride_w, size_t pad_h, size_t pad_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const size_t input_offset = (n * channels + c) * input_h * input_w;
    const size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid =
            std::min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
        long w_end_valid =
            std::min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

        T sum = T(0);

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            sum += input_data[input_offset + ih * input_w + iw];
          }
        }

        const size_t output_idx = output_offset + out_h * output_w + out_w;
        output_data[output_idx] = sum * pool_size_inv;
      }
    }
  });
}

template <typename T>
void compute_avg_pool_backward(const T *gradient_data, T *grad_input_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const size_t input_offset = (n * channels + c) * input_h * input_w;
    const size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        const size_t output_idx = output_offset + out_h * output_w + out_w;

        const T grad_val = gradient_data[output_idx] * pool_size_inv;

        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid =
            std::min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
        long w_end_valid =
            std::min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            grad_input_data[input_offset + ih * input_w + iw] += grad_val;
          }
        }
      }
    }
  });
}

#define INSTANTIATE_AVGPOOL(T)                                                                 \
  template void compute_avg_pool_forward<T>(                                                   \
      const T *input_data, T *output_data, size_t batch_size, size_t channels, size_t input_h, \
      size_t input_w, size_t output_h, size_t output_w, size_t pool_h, size_t pool_w,          \
      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);                           \
                                                                                               \
  template void compute_avg_pool_backward<T>(                                                  \
      const T *gradient_data, T *grad_input_data, size_t batch_size, size_t channels,          \
      size_t input_h, size_t input_w, size_t output_h, size_t output_w, size_t pool_h,         \
      size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);

INSTANTIATE_AVGPOOL(fp16)
INSTANTIATE_AVGPOOL(bf16)
INSTANTIATE_AVGPOOL(float)
INSTANTIATE_AVGPOOL(double)
#undef INSTANTIATE_AVGPOOL

}  // namespace avgpool_nchw
}  // namespace cpu
}  // namespace tnn
