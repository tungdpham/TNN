/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/maxpool_ops.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "type/type.hpp"

namespace tnn {
namespace cpu {
template <typename T>
void maxpool_forward(const T *input, T *output, int *mask_indices, size_t batch_size, size_t height,
                     size_t width, size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                     size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                     size_t output_w) {
  std::vector<float> local_max(channels);
  std::vector<int> local_idx(channels);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      int h_start = std::max(static_cast<int>(oh * stride_h) - static_cast<int>(pad_h), 0);
      int h_end = std::min(
          static_cast<int>(oh * stride_h) - static_cast<int>(pad_h) + static_cast<int>(pool_h),
          static_cast<int>(height));

      for (size_t ow = 0; ow < output_w; ++ow) {
        int w_start = std::max(static_cast<int>(ow * stride_w) - static_cast<int>(pad_w), 0);
        int w_end = std::min(
            static_cast<int>(ow * stride_w) - static_cast<int>(pad_w) + static_cast<int>(pool_w),
            static_cast<int>(width));

        for (size_t c = 0; c < channels; ++c) {
          local_max[c] = -std::numeric_limits<float>::infinity();
          local_idx[c] = -1;
        }

        for (int h = h_start; h < h_end; ++h) {
          for (int w = w_start; w < w_end; ++w) {
            size_t base_input_idx = ((b * height + h) * width + w) * channels;

            for (size_t c = 0; c < channels; ++c) {
              size_t input_idx = base_input_idx + c;
              float val = static_cast<float>(input[input_idx]);

              if (val > local_max[c]) {
                local_max[c] = val;
                local_idx[c] = static_cast<int>(input_idx);
              }
            }
          }
        }

        size_t base_output_idx = ((b * output_h + oh) * output_w + ow) * channels;
        for (size_t c = 0; c < channels; ++c) {
          output[base_output_idx + c] = static_cast<T>(local_max[c]);
          mask_indices[base_output_idx + c] = local_idx[c];
        }
      }
    }
  }
}

template <typename T>
void maxpool_backward(const T *grad_output, T *grad_input, const int *mask_indices,
                      size_t batch_size, size_t channels, size_t output_h, size_t output_w) {
  size_t total_outputs = batch_size * output_h * output_w * channels;
  for (size_t i = 0; i < total_outputs; ++i) {
    int max_idx = mask_indices[i];
    if (max_idx >= 0) {
      grad_input[max_idx] = static_cast<T>(static_cast<float>(grad_input[max_idx]) +
                                           static_cast<float>(grad_output[i]));
    }
  }
}

#define INSTANTIATE_MAXPOOL(T)                                                                    \
  template void maxpool_forward<T>(                                                               \
      const T *input, T *output, int *mask_indices, size_t batch_size, size_t height,             \
      size_t width, size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,               \
      size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);             \
  template void maxpool_backward<T>(const T *grad_output, T *grad_input, const int *mask_indices, \
                                    size_t batch_size, size_t channels, size_t output_h,          \
                                    size_t output_w);

INSTANTIATE_MAXPOOL(fp16)
INSTANTIATE_MAXPOOL(bf16)
INSTANTIATE_MAXPOOL(float)
INSTANTIATE_MAXPOOL(double)

#undef INSTANTIATE_MAXPOOL

}  // namespace cpu
}  // namespace tnn
