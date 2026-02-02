/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <immintrin.h>

#include "ops/cpu/kernels.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {
/**
 * CPU tensor operations that work with raw pointers and dimensions.
 * These functions handle the actual computation without device checks or shape extraction.
 * All high-level logic (shape validation, padding calculations, etc.) should be done at the caller
 * level.
 */

// Specialized optimized implementation for 3x3 kernel with padding=1 and stride=1
template <typename T>
void cpu_im2col_pad_1_stride_1_kernel_3(const T *input_data, T *col_data, size_t batch_size,
                                        size_t channels, size_t height, size_t width,
                                        size_t output_h, size_t output_w) {
  size_t col_width = output_h * output_w;

  const __m256 zero = _mm256_setzero_ps();

  if constexpr (std::is_same_v<T, float>) {
    parallel_for_2d<size_t>(
        batch_size, channels,
        [&](size_t n, size_t c) {
          const float *input_channel_ptr = input_data + (n * channels + c) * height * width;
          const size_t batch_offset = n * col_width;
          const size_t col_stride = batch_size * col_width;
          constexpr size_t simd_width = 8;

          const size_t simd_end_full = (width >> 3) << 3;
          const size_t simd_end_minus2 = ((width - 2) >> 3) << 3;

          // kh=0: Process all 3 kw values together (top row)
          {
            size_t col_row_idx_base = c * 9 + 0;
            float *col_row_kw0 = col_data + (col_row_idx_base + 0) * col_stride + batch_offset;
            float *col_row_kw1 = col_data + (col_row_idx_base + 1) * col_stride + batch_offset;
            float *col_row_kw2 = col_data + (col_row_idx_base + 2) * col_stride + batch_offset;

            for (size_t i = 0; i < simd_end_full; i += simd_width) {
              _mm256_storeu_ps(col_row_kw0 + i, zero);
              _mm256_storeu_ps(col_row_kw1 + i, zero);
              _mm256_storeu_ps(col_row_kw2 + i, zero);
            }
            for (size_t i = simd_end_full; i < width; ++i) {
              col_row_kw0[i] = 0.0f;
              col_row_kw1[i] = 0.0f;
              col_row_kw2[i] = 0.0f;
            }

            const float *input_ptr = input_channel_ptr;
            float *col_ptr_kw0 = col_row_kw0 + width;
            float *col_ptr_kw1 = col_row_kw1 + width;
            float *col_ptr_kw2 = col_row_kw2 + width;

            for (size_t h = 1; h < height; ++h) {
              col_ptr_kw0[0] = 0.0f;

              for (size_t w = 0; w < simd_end_minus2; w += simd_width) {
                __m256 data0 = _mm256_loadu_ps(input_ptr + w);
                __m256 data1 = _mm256_loadu_ps(input_ptr + w + 1);

                _mm256_storeu_ps(col_ptr_kw0 + 1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw2 + w, data1);
              }

              for (size_t w = simd_end_minus2; w < width - 1; ++w) {
                float val = input_ptr[w];
                col_ptr_kw0[1 + w] = val;
                col_ptr_kw1[w] = val;
                col_ptr_kw2[w] = input_ptr[w + 1];
              }

              col_ptr_kw2[width - 1] = 0.0f;
              col_ptr_kw1[width - 1] = input_ptr[width - 1];

              input_ptr += width;
              col_ptr_kw0 += width;
              col_ptr_kw1 += width;
              col_ptr_kw2 += width;
            }
          }

          // kh=1: Process all 3 kw values together (middle row)
          {
            size_t col_row_idx_base = c * 9 + 3;
            float *col_row_kw0 = col_data + (col_row_idx_base + 0) * col_stride + batch_offset;
            float *col_row_kw1 = col_data + (col_row_idx_base + 1) * col_stride + batch_offset;
            float *col_row_kw2 = col_data + (col_row_idx_base + 2) * col_stride + batch_offset;

            const float *input_ptr = input_channel_ptr;
            float *col_ptr_kw0 = col_row_kw0;
            float *col_ptr_kw1 = col_row_kw1;
            float *col_ptr_kw2 = col_row_kw2;

            for (size_t h = 0; h < height; ++h) {
              col_ptr_kw0[0] = 0.0f;

              for (size_t w = 0; w < simd_end_minus2; w += simd_width) {
                __m256 data0 = _mm256_loadu_ps(input_ptr + w);
                __m256 data1 = _mm256_loadu_ps(input_ptr + w + 1);

                _mm256_storeu_ps(col_ptr_kw0 + 1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw2 + w, data1);
              }

              for (size_t w = simd_end_minus2; w < width - 1; ++w) {
                float val = input_ptr[w];
                col_ptr_kw0[1 + w] = val;
                col_ptr_kw1[w] = val;
                col_ptr_kw2[w] = input_ptr[w + 1];
              }

              col_ptr_kw2[width - 1] = 0.0f;
              col_ptr_kw1[width - 1] = input_ptr[width - 1];

              input_ptr += width;
              col_ptr_kw0 += width;
              col_ptr_kw1 += width;
              col_ptr_kw2 += width;
            }
          }

          // kh=2: Process all 3 kw values together (bottom row)
          {
            size_t col_row_idx_base = c * 9 + 6;
            float *col_row_kw0 = col_data + (col_row_idx_base + 0) * col_stride + batch_offset;
            float *col_row_kw1 = col_data + (col_row_idx_base + 1) * col_stride + batch_offset;
            float *col_row_kw2 = col_data + (col_row_idx_base + 2) * col_stride + batch_offset;

            const float *input_ptr = input_channel_ptr + width;
            float *col_ptr_kw0 = col_row_kw0;
            float *col_ptr_kw1 = col_row_kw1;
            float *col_ptr_kw2 = col_row_kw2;

            for (size_t h = 0; h < height - 1; ++h) {
              col_ptr_kw0[0] = 0.0f;

              for (size_t w = 0; w < simd_end_minus2; w += simd_width) {
                __m256 data0 = _mm256_loadu_ps(input_ptr + w);
                __m256 data1 = _mm256_loadu_ps(input_ptr + w + 1);

                _mm256_storeu_ps(col_ptr_kw0 + 1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw2 + w, data1);
              }

              for (size_t w = simd_end_minus2; w < width - 1; ++w) {
                float val = input_ptr[w];
                col_ptr_kw0[1 + w] = val;
                col_ptr_kw1[w] = val;
                col_ptr_kw2[w] = input_ptr[w + 1];
              }

              col_ptr_kw2[width - 1] = 0.0f;
              col_ptr_kw1[width - 1] = input_ptr[width - 1];

              input_ptr += width;
              col_ptr_kw0 += width;
              col_ptr_kw1 += width;
              col_ptr_kw2 += width;
            }

            for (size_t i = 0; i < simd_end_full; i += simd_width) {
              _mm256_storeu_ps(col_ptr_kw0 + i, zero);
              _mm256_storeu_ps(col_ptr_kw1 + i, zero);
              _mm256_storeu_ps(col_ptr_kw2 + i, zero);
            }
            for (size_t i = simd_end_full; i < width; ++i) {
              col_ptr_kw0[i] = 0.0f;
              col_ptr_kw1[i] = 0.0f;
              col_ptr_kw2[i] = 0.0f;
            }
          }
        },
        SchedulePolicy::Auto);
  } else {
    // TODO: Add non-float implementation if needed
  }
}

template <typename T>
void cpu_im2col(const T *input_data, T *col_data, size_t batch_size, size_t channels, size_t height,
                size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                size_t pad_h, size_t pad_w, size_t output_h, size_t output_w) {
  size_t col_width = output_h * output_w;

  parallel_for_2d<size_t>(
      batch_size, channels,
      [&](size_t n, size_t c) {
        const T *input_channel_ptr = input_data + (n * channels + c) * height * width;
        const size_t batch_offset = n * col_width;
        const size_t col_stride = batch_size * col_width;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            T *col_row_base = col_data + col_row_idx * col_stride + batch_offset;

            const size_t h_start = (pad_h > kh) ? ((pad_h - kh + stride_h - 1) / stride_h) : 0;
            const size_t h_end =
                std::min(output_h, (height + pad_h - kh + stride_h - 1) / stride_h);
            const size_t w_start = (pad_w > kw) ? ((pad_w - kw + stride_w - 1) / stride_w) : 0;
            const size_t w_end = std::min(output_w, (width + pad_w - kw + stride_w - 1) / stride_w);

            std::fill(col_row_base, col_row_base + h_start * output_w, T(0));

            for (size_t h = h_start; h < h_end; ++h) {
              const size_t h_in = h * stride_h + kh - pad_h;
              const T *input_row = input_channel_ptr + h_in * width;
              T *col_ptr = col_row_base + h * output_w;

              std::fill(col_ptr, col_ptr + w_start, T(0));

              if (stride_w == 1) {
                const size_t w_in_start = w_start + kw - pad_w;
                const size_t copy_len = w_end - w_start;
                std::copy(input_row + w_in_start, input_row + w_in_start + copy_len,
                          col_ptr + w_start);
              } else {
                for (size_t w = w_start; w < w_end; ++w) {
                  const size_t w_in = w * stride_w + kw - pad_w;
                  col_ptr[w] = input_row[w_in];
                }
              }

              std::fill(col_ptr + w_end, col_ptr + output_w, T(0));
            }

            std::fill(col_row_base + h_end * output_w, col_row_base + output_h * output_w, T(0));
          }
        }
      },
      SchedulePolicy::Auto);
}

template <typename T>
void cpu_col2im(const T *col_data, T *result_data, size_t batch_size, size_t channels,
                size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
                size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h, size_t output_w) {
  const size_t col_width = output_h * output_w;

  // Initialize result buffer to zero
  std::fill(result_data, result_data + batch_size * channels * height * width, T(0));

  parallel_for_2d<size_t>(
      batch_size, channels,
      [&](size_t n, size_t c) {
        T *result_channel_ptr = result_data + (n * channels + c) * height * width;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            const T *col_row_ptr =
                col_data + col_row_idx * (batch_size * col_width) + n * col_width;

            const size_t h_start = pad_h > kh ? ((pad_h - kh + stride_h - 1) / stride_h) : 0;
            const size_t h_end =
                std::min(output_h, (height + pad_h - kh + stride_h - 1) / stride_h);

            const size_t w_start = pad_w > kw ? ((pad_w - kw + stride_w - 1) / stride_w) : 0;
            const size_t w_end = std::min(output_w, (width + pad_w - kw + stride_w - 1) / stride_w);

            col_row_ptr += h_start * output_w;

            for (size_t h = h_start; h < h_end; ++h) {
              const size_t h_out = h * stride_h + kh - pad_h;
              T *result_row = result_channel_ptr + h_out * width;

              col_row_ptr += w_start;

              size_t valid_width = w_end - w_start;
              if (stride_w == 1) {
                size_t w_out_start = w_start * stride_w + kw - pad_w;
                ops::cpu::add(result_row + w_out_start, col_row_ptr, result_row + w_out_start,
                              valid_width);
                col_row_ptr += valid_width;
              } else {
                for (size_t w = w_start; w < w_end; ++w) {
                  const size_t w_out = w * stride_w + kw - pad_w;
                  result_row[w_out] += *col_row_ptr++;
                }
              }

              col_row_ptr += (output_w - w_end);
            }
            col_row_ptr += (output_h - h_end) * output_w;
          }
        }
      },
      SchedulePolicy::Auto);
}

template <typename T>
void cpu_pad(const T *input_data, T *result_data, size_t batch_size, size_t channels, size_t height,
             size_t width, size_t pad_h, size_t pad_w, T value) {
  const size_t padded_height = height + 2 * pad_h;
  const size_t padded_width = width + 2 * pad_w;

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    // fill top padding rows
    for (size_t h = 0; h < pad_h; ++h) {
      std::fill(
          &result_data[((n * channels + c) * padded_height + h) * padded_width],
          &result_data[((n * channels + c) * padded_height + h) * padded_width] + padded_width,
          value);
    }

    // Copy middle rows with left and right padding
    for (size_t h = 0; h < height; ++h) {
      const size_t new_h = h + pad_h;
      // copy the row over
      std::copy(&input_data[((n * channels + c) * height + h) * width],
                &input_data[((n * channels + c) * height + h) * width] + width,
                &result_data[((n * channels + c) * padded_height + new_h) * padded_width + pad_w]);

      // set values on left and right
      std::fill(&result_data[((n * channels + c) * padded_height + new_h) * padded_width],
                &result_data[((n * channels + c) * padded_height + new_h) * padded_width + pad_w],
                value);

      // right side
      std::fill(
          &result_data[((n * channels + c) * padded_height + new_h) * padded_width + pad_w + width],
          &result_data[((n * channels + c) * padded_height + new_h) * padded_width + padded_width],
          value);
    }

    // fill bottom padding rows
    for (size_t h = height + pad_h; h < padded_height; ++h) {
      std::fill(
          &result_data[((n * channels + c) * padded_height + h) * padded_width],
          &result_data[((n * channels + c) * padded_height + h) * padded_width] + padded_width,
          value);
    }
  });
}

template <typename T>
void cpu_unpad(const T *input_data, T *result_data, size_t batch_size, size_t channels,
               size_t height, size_t width, size_t pad_h, size_t pad_w) {
  const size_t padded_height = height + 2 * pad_h;
  const size_t padded_width = width + 2 * pad_w;

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t h = 0; h < height; ++h) {
      const size_t src_h = h + pad_h;
      std::copy(
          &input_data[((n * channels + c) * padded_height + src_h) * padded_width + pad_w],
          &input_data[((n * channels + c) * padded_height + src_h) * padded_width + pad_w] + width,
          &result_data[((n * channels + c) * height + h) * width]);
    }
  });
}

template <typename T>
void cpu_crop(const T *input_data, T *result_data, size_t batch_size, size_t channels,
              size_t height, size_t width, size_t start_h, size_t start_w, size_t new_height,
              size_t new_width) {
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < new_height; ++h) {
        std::copy(&input_data[((n * channels + c) * height + (h + start_h)) * width + start_w],
                  &input_data[((n * channels + c) * height + (h + start_h)) * width + start_w] +
                      new_width,
                  &result_data[((n * channels + c) * new_height + h) * new_width]);
      }
    }
  }
}

template <typename T>
void cpu_transpose_2d(const T *src, T *dst, const size_t rows, const size_t cols) {
  const size_t block_size = 64;
  parallel_for_2d((rows + block_size - 1) / block_size, (cols + block_size - 1) / block_size,
                  [&](size_t i_block, size_t j_block) {
                    const size_t start_row = i_block * block_size;
                    const size_t start_col = j_block * block_size;
                    const size_t end_row = std::min(start_row + block_size, rows);
                    const size_t end_col = std::min(start_col + block_size, cols);
                    for (size_t i = start_row; i < end_row; ++i) {
                      for (size_t j = start_col; j < end_col; ++j) {
                        dst[j * rows + i] = src[i * cols + j];
                      }
                    }
                  });
}

template <typename T>
void cpu_nchw_to_cnhw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                      size_t width) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[n * channels * height * width + c * height * width],
              &src[n * channels * height * width + c * height * width + height * width],
              &dst[c * batch_size * height * width + n * height * width]);
  });
}

template <typename T>
void cpu_cnhw_to_nchw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                      size_t width) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[c * batch_size * height * width + n * height * width],
              &src[c * batch_size * height * width + n * height * width + height * width],
              &dst[n * channels * height * width + c * height * width]);
  });
}

/**
 * @brief Slice the tensor along the batch dimension.
 * @param input_data Input tensor data pointer
 * @param result_data Output tensor data pointer
 * @param batch_size Total batch size
 * @param start_batch Starting batch index (inclusive)
 * @param end_batch Ending batch index (exclusive)
 * @param stride_0 Stride for the batch dimension (product of all dimensions except batch)
 */
template <typename T>
void cpu_slice_batch(const T *input_data, T *result_data, size_t start_batch, size_t end_batch,
                     size_t stride_0) {
  std::copy(&input_data[start_batch * stride_0], &input_data[end_batch * stride_0], result_data);
}

/**
 * @brief Split the tensor into multiple parts along the batch dimension.
 * @param input_data Input tensor data pointer
 * @param results Vector of output data pointers
 * @param batch_size Total batch size
 * @param num_splits Number of splits to create
 * @param stride_0 Stride for the batch dimension (product of all dimensions except batch)
 */
template <typename T>
void cpu_split(const T *input_data, std::vector<T *> &results, size_t batch_size, size_t num_splits,
               size_t stride_0) {
  size_t split_size = batch_size / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? batch_size : start + split_size;
    cpu_slice_batch<T>(input_data, results[i], start, end, stride_0);
  }
}

}  // namespace cpu
}  // namespace tnn
