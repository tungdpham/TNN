/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/avgpool_ops.hpp"
#include "type/type.hpp"

#include <algorithm>
#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T>
void avgpool_forward(const T *input, T *output, size_t batch_size, size_t height, size_t width,
                     size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                     size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                     size_t output_w) {
  // NHWC format: [batch, height, width, channels]
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t c = 0; c < channels; ++c) {
          float sum = 0.0f;
          int count = 0;

          // Calculate input coordinates
          int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
          int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);

          // Average pooling over the window
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              size_t input_idx = ((b * height + h) * width + w) * channels + c;
              sum += static_cast<float>(input[input_idx]);
              ++count;
            }
          }

          // Store average
          size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
          output[output_idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
        }
      }
    }
  }
}

template <typename T>
void avgpool_backward(const T *grad_output, T *grad_input, size_t batch_size, size_t input_h,
                      size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  // NHWC format: [batch, height, width, channels]
  // grad_input should already be zeroed
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t c = 0; c < channels; ++c) {
          size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
          float grad = static_cast<float>(grad_output[output_idx]);

          int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
          int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);

          int count = (h_end - h_start) * (w_end - w_start);
          if (count == 0)
            continue;

          float grad_per_element = grad / count;
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
              grad_input[input_idx] =
                  static_cast<T>(static_cast<float>(grad_input[input_idx]) + grad_per_element);
            }
          }
        }
      }
    }
  }
}

#define INSTANTIATE_AVGPOOL(T)                                                                     \
  template void avgpool_forward<T>(const T *input, T *output, size_t batch_size, size_t height,    \
                                   size_t width, size_t channels, size_t pool_h, size_t pool_w,    \
                                   size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,   \
                                   size_t output_h, size_t output_w);                              \
  template void avgpool_backward<T>(                                                               \
      const T *grad_output, T *grad_input, size_t batch_size, size_t input_h, size_t input_w,      \
      size_t channels, size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,             \
      size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);

INSTANTIATE_AVGPOOL(fp16)
INSTANTIATE_AVGPOOL(bf16)
INSTANTIATE_AVGPOOL(float)
INSTANTIATE_AVGPOOL(double)

#undef INSTANTIATE_AVGPOOL

} // namespace cpu
} // namespace tnn
