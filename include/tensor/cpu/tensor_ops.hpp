/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "matrix/matrix.hpp"
#include "ops/cpu/kernels.hpp"
#include "tensor/tensor.hpp"
#include <immintrin.h>

namespace tnn {
namespace cpu {
/**
 * NOTE: These im2col/col2im implementations are CPU-only and heavily optimized with AVX2.
 * For GPU tensors, use the GPU-aware versions in gpu namespace.
 * All functions in this file expect CPU tensors and will throw if given GPU tensors.
 */
template <typename T>
void im2col_pad_1_stride_1_kernel_3(const Tensor<T, NCHW> &input, T *col_data) {
  if (!input.is_on_cpu()) {
    throw std::runtime_error("im2col_pad_1_stride_1_kernel_3_cpu requires CPU tensor");
  }

  const size_t in_h = input.height();
  const size_t in_w = input.width();
  const size_t channels = input.channels();
  const size_t batch_size = input.batch_size();

  size_t col_width = in_h * in_w;

  const __m256 zero = _mm256_setzero_ps();

  const T *input_data = input.data_ptr().get();

  if constexpr (std::is_same_v<T, float>) {
    parallel_for_2d<size_t>(
        batch_size, channels,
        [&](size_t n, size_t c) {
          const float *input_channel_ptr = input_data + (n * channels + c) * in_h * in_w;
          const size_t batch_offset = n * col_width;
          const size_t col_stride = batch_size * col_width;
          constexpr size_t simd_width = 8;

          const size_t simd_end_full = (in_w >> 3) << 3;
          const size_t simd_end_minus2 = ((in_w - 2) >> 3) << 3;

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
            for (size_t i = simd_end_full; i < in_w; ++i) {
              col_row_kw0[i] = 0.0f;
              col_row_kw1[i] = 0.0f;
              col_row_kw2[i] = 0.0f;
            }

            const float *input_ptr = input_channel_ptr;
            float *col_ptr_kw0 = col_row_kw0 + in_w;
            float *col_ptr_kw1 = col_row_kw1 + in_w;
            float *col_ptr_kw2 = col_row_kw2 + in_w;

            for (size_t h = 1; h < in_h; ++h) {
              col_ptr_kw0[0] = 0.0f;

              for (size_t w = 0; w < simd_end_minus2; w += simd_width) {
                __m256 data0 = _mm256_loadu_ps(input_ptr + w);
                __m256 data1 = _mm256_loadu_ps(input_ptr + w + 1);

                _mm256_storeu_ps(col_ptr_kw0 + 1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw2 + w, data1);
              }

              for (size_t w = simd_end_minus2; w < in_w - 1; ++w) {
                float val = input_ptr[w];
                col_ptr_kw0[1 + w] = val;
                col_ptr_kw1[w] = val;
                col_ptr_kw2[w] = input_ptr[w + 1];
              }

              col_ptr_kw2[in_w - 1] = 0.0f;
              col_ptr_kw1[in_w - 1] = input_ptr[in_w - 1];

              input_ptr += in_w;
              col_ptr_kw0 += in_w;
              col_ptr_kw1 += in_w;
              col_ptr_kw2 += in_w;
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

            for (size_t h = 0; h < in_h; ++h) {
              col_ptr_kw0[0] = 0.0f;

              for (size_t w = 0; w < simd_end_minus2; w += simd_width) {
                __m256 data0 = _mm256_loadu_ps(input_ptr + w);
                __m256 data1 = _mm256_loadu_ps(input_ptr + w + 1);

                _mm256_storeu_ps(col_ptr_kw0 + 1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw2 + w, data1);
              }

              for (size_t w = simd_end_minus2; w < in_w - 1; ++w) {
                float val = input_ptr[w];
                col_ptr_kw0[1 + w] = val;
                col_ptr_kw1[w] = val;
                col_ptr_kw2[w] = input_ptr[w + 1];
              }

              col_ptr_kw2[in_w - 1] = 0.0f;
              col_ptr_kw1[in_w - 1] = input_ptr[in_w - 1];

              input_ptr += in_w;
              col_ptr_kw0 += in_w;
              col_ptr_kw1 += in_w;
              col_ptr_kw2 += in_w;
            }
          }

          // kh=2: Process all 3 kw values together (bottom row)
          {
            size_t col_row_idx_base = c * 9 + 6;
            float *col_row_kw0 = col_data + (col_row_idx_base + 0) * col_stride + batch_offset;
            float *col_row_kw1 = col_data + (col_row_idx_base + 1) * col_stride + batch_offset;
            float *col_row_kw2 = col_data + (col_row_idx_base + 2) * col_stride + batch_offset;

            const float *input_ptr = input_channel_ptr + in_w;
            float *col_ptr_kw0 = col_row_kw0;
            float *col_ptr_kw1 = col_row_kw1;
            float *col_ptr_kw2 = col_row_kw2;

            for (size_t h = 0; h < in_h - 1; ++h) {
              col_ptr_kw0[0] = 0.0f;

              for (size_t w = 0; w < simd_end_minus2; w += simd_width) {
                __m256 data0 = _mm256_loadu_ps(input_ptr + w);
                __m256 data1 = _mm256_loadu_ps(input_ptr + w + 1);

                _mm256_storeu_ps(col_ptr_kw0 + 1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw1 + w, data0);
                _mm256_storeu_ps(col_ptr_kw2 + w, data1);
              }

              // Scalar cleanup - FIXED
              for (size_t w = simd_end_minus2; w < in_w - 1; ++w) {
                float val = input_ptr[w];
                col_ptr_kw0[1 + w] = val;
                col_ptr_kw1[w] = val;
                col_ptr_kw2[w] = input_ptr[w + 1];
              }

              col_ptr_kw2[in_w - 1] = 0.0f;
              col_ptr_kw1[in_w - 1] = input_ptr[in_w - 1];

              input_ptr += in_w;
              col_ptr_kw0 += in_w;
              col_ptr_kw1 += in_w;
              col_ptr_kw2 += in_w;
            }

            for (size_t i = 0; i < simd_end_full; i += simd_width) {
              _mm256_storeu_ps(col_ptr_kw0 + i, zero);
              _mm256_storeu_ps(col_ptr_kw1 + i, zero);
              _mm256_storeu_ps(col_ptr_kw2 + i, zero);
            }
            for (size_t i = simd_end_full; i < in_w; ++i) {
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
void im2col_padded(const Tensor<T, NCHW> &input_tensor, T *col_data, const size_t kernel_h,
                   const size_t kernel_w, const size_t stride_h, const size_t stride_w,
                   const size_t pad_h, const size_t pad_w) {
  if (!input_tensor.is_on_cpu()) {
    throw std::runtime_error("im2col_padded_cpu requires CPU tensor");
  }

  const size_t in_h = input_tensor.height();
  const size_t in_w = input_tensor.width();
  const size_t padded_h = in_h + 2 * pad_h;
  const size_t padded_w = in_w + 2 * pad_w;
  const size_t out_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t out_w = (padded_w - kernel_w) / stride_w + 1;
  const size_t channels = input_tensor.channels();
  const size_t batch_size = input_tensor.batch_size();

  size_t col_width = out_h * out_w;

  const T *input_data = input_tensor.data_ptr().get();

  parallel_for_2d<size_t>(
      batch_size, channels,
      [&](size_t n, size_t c) {
        const T *input_channel_ptr = input_data + (n * channels + c) * in_h * in_w;
        const size_t batch_offset = n * col_width;
        const size_t col_stride = batch_size * col_width;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            T *col_row_base = col_data + col_row_idx * col_stride + batch_offset;

            const size_t h_start = (pad_h > kh) ? ((pad_h - kh + stride_h - 1) / stride_h) : 0;
            const size_t h_end = std::min(out_h, (in_h + pad_h - kh + stride_h - 1) / stride_h);
            const size_t w_start = (pad_w > kw) ? ((pad_w - kw + stride_w - 1) / stride_w) : 0;
            const size_t w_end = std::min(out_w, (in_w + pad_w - kw + stride_w - 1) / stride_w);

            std::fill(col_row_base, col_row_base + h_start * out_w, T(0));

            for (size_t h = h_start; h < h_end; ++h) {
              const size_t h_in = h * stride_h + kh - pad_h;
              const T *input_row = input_channel_ptr + h_in * in_w;
              T *col_ptr = col_row_base + h * out_w;

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

              std::fill(col_ptr + w_end, col_ptr + out_w, T(0));
            }

            std::fill(col_row_base + h_end * out_w, col_row_base + out_h * out_w, T(0));
          }
        }
      },
      SchedulePolicy::Auto);
}

/**
 * @brief Convert a 4D image tensor to a column buffer for convolution (raw pointer version).
 * @param input_tensor The input tensor to convert.
 * @param col_data Pointer to the output column buffer.
 * @param kernel_h Height of the convolution kernel.
 * @param kernel_w Width of the convolution kernel.
 * @param stride_h Vertical stride of the convolution.
 * @param stride_w Horizontal stride of the convolution.
 * @param pad_h Vertical padding to be applied to the input tensor.
 * @param pad_w Horizontal padding to be applied to the input tensor.
 */
template <typename T>
void im2col(const Tensor<T, NCHW> &input_tensor, T *col_data, size_t kernel_h, size_t kernel_w,
            size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0) {
  if (!input_tensor.is_on_cpu()) {
    throw std::runtime_error("im2col_cpu requires CPU tensor");
  }

  if (pad_h > 0 || pad_w > 0) {
    if (pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1 && kernel_h == 3 &&
        kernel_w == 3)
      im2col_pad_1_stride_1_kernel_3(input_tensor, col_data);
    else
      im2col_padded(input_tensor, col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    return;
  }

  const size_t in_h = input_tensor.height();
  const size_t in_w = input_tensor.width();
  const size_t out_h = (in_h - kernel_h) / stride_h + 1;
  const size_t out_w = (in_w - kernel_w) / stride_w + 1;
  const size_t channels = input_tensor.channels();
  const size_t batch_size = input_tensor.batch_size();

  size_t col_width = out_h * out_w;

  const T *input_data = input_tensor.data_ptr().get();

  parallel_for_2d<size_t>(
      batch_size, channels,
      [&](size_t n, size_t c) {
        const T *input_channel_ptr = input_data + (n * channels + c) * in_h * in_w;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            T *col_row_ptr = col_data + col_row_idx * (batch_size * col_width) + n * col_width;

            for (size_t h = 0; h < out_h; ++h) {
              const T *input_row_ptr = input_channel_ptr + (h * stride_h + kh) * in_w + kw;

              //  vectorize where possible
              if (stride_w == 1) {
                std::copy(input_row_ptr, input_row_ptr + out_w, col_row_ptr);
                col_row_ptr += out_w;
              } else {
                for (size_t w = 0; w < out_w; ++w) {
                  *col_row_ptr++ = input_row_ptr[w * stride_w];
                }
              }
            }
          }
        }
      },
      SchedulePolicy::Auto);
}

template <typename T>
static void col2im_padded(const T *col_data, T *result_data, size_t batch_size, size_t channels,
                          size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                          size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
  const size_t padded_h = height + 2 * pad_h;
  const size_t padded_w = width + 2 * pad_w;
  const size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t output_w = (padded_w - kernel_w) / stride_w + 1;
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

/**
 * @brief Convert a column buffer back to the original image tensor (raw pointer version).
 * @param col_data The input column buffer.
 * @param result_data The output tensor buffer.
 * @param batch_size Number of images in the batch.
 * @param channels Number of channels in the images.
 * @param height Height of the original images.
 * @param width Width of the original images.
 * @param kernel_h Height of the convolution kernel.
 * @param kernel_w Width of the convolution kernel.
 * @param stride_h Vertical stride of the convolution.
 * @param stride_w Horizontal stride of the convolution.
 * @param pad_h Vertical padding applied to the original images.
 * @param pad_w Horizontal padding applied to the original images.
 */
template <typename T>
static void col2im(const T *col_data, T *result_data, size_t batch_size, size_t channels,
                   size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
                   size_t stride_w, size_t pad_h, size_t pad_w) {
  if (pad_h > 0 || pad_w > 0) {
    col2im_padded(col_data, result_data, batch_size, channels, height, width, kernel_h, kernel_w,
                  stride_h, stride_w, pad_h, pad_w);
    return;
  }

  // No padding case
  size_t output_h = (height - kernel_h) / stride_h + 1;
  size_t output_w = (width - kernel_w) / stride_w + 1;

  const size_t col_width = output_h * output_w;

  parallel_for_2d<size_t>(
      batch_size, channels,
      [&](size_t n, size_t c) {
        T *result_channel_ptr = result_data + (n * channels + c) * height * width;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            const T *col_row_ptr =
                col_data + col_row_idx * (batch_size * col_width) + n * col_width;

            for (size_t h = 0; h < output_h; ++h) {
              T *result_row_ptr = result_channel_ptr + (h * stride_h + kh) * width + kw;

              if (stride_w == 1) {
                ops::cpu::add(result_row_ptr, col_row_ptr, result_row_ptr, output_w);
                col_row_ptr += output_w;
              } else {
                for (size_t w = 0; w < output_w; ++w) {
                  result_row_ptr[w * stride_w] += *col_row_ptr++;
                }
              }
            }
          }
        }
      },
      SchedulePolicy::Static);
}

template <typename T, Layout L>
void pad(const Tensor<T, L> &input, Tensor<T, L> &result, size_t pad_h, size_t pad_w,
         T value = T(0)) {
  throw std::runtime_error("Unsupported tensor layout for padding");
}

template <typename T>
void pad(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t pad_h, size_t pad_w,
         T value = T(0)) {
  if (!input.is_on_cpu()) {
    throw std::runtime_error("pad requires CPU tensor");
  }

  const size_t batch_size_ = input.batch_size();
  const size_t channels_ = input.channels();
  const size_t height_ = input.height();
  const size_t width_ = input.width();

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
    const size_t padded_height = height_ + 2 * pad_h;
    const size_t padded_width = width_ + 2 * pad_w;
    // fill top padding rows
    for (size_t h = 0; h < pad_h; ++h) {
      std::fill(&result_data[((n * channels_ + c) * padded_height + h) * padded_width],
                &result_data[((n * channels_ + c) * padded_height + h) * padded_width] +
                    padded_width,
                value);
    }

    // Copy middle rows with left and right padding
    for (size_t h = 0; h < height_; ++h) {
      const size_t new_h = h + pad_h;
      // copy the row over
      std::copy(&input_data[((n * channels_ + c) * height_ + h) * width_],
                &input_data[((n * channels_ + c) * height_ + h) * width_] + width_,
                &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w]);

      // set values on left and right
      std::fill(&result_data[((n * channels_ + c) * padded_height + new_h) * padded_width],
                &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w],
                value);

      // right side
      std::fill(
          &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w +
                       width_],
          &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + padded_width],
          value);
    }

    // fill bottom padding rows
    for (size_t h = height_ + pad_h; h < padded_height; ++h) {
      std::fill(&result_data[((n * channels_ + c) * padded_height + h) * padded_width],
                &result_data[((n * channels_ + c) * padded_height + h) * padded_width] +
                    padded_width,
                value);
    }
  });
}

template <typename T, Layout L>
void unpad(const Tensor<T, L> &input, Tensor<T, NCHW> &result, size_t pad_h, size_t pad_w) {
  throw std::runtime_error("Unsupported tensor layout for unpadding");
}

template <typename T>
void unpad(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t pad_h, size_t pad_w) {
  if (!input.is_on_cpu()) {
    throw std::runtime_error("unpad requires CPU tensor");
  }

  const size_t batch_size_ = input.batch_size();
  const size_t channels_ = input.channels();
  const size_t height_ = input.height();
  const size_t width_ = input.width();

  if (height_ <= 2 * pad_h || width_ <= 2 * pad_w) {
    throw std::invalid_argument("Padding size too large for unpadding");
  }

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
    for (size_t h = 0; h < height_ - 2 * pad_h; ++h) {
      const size_t src_h = h + pad_h;
      std::copy(
          &input_data[((n * channels_ + c) * height_ + src_h) * width_ + pad_w],
          &input_data[((n * channels_ + c) * height_ + src_h) * width_ + pad_w] +
              (width_ - 2 * pad_w),
          &result_data[((n * channels_ + c) * (height_ - 2 * pad_h) + h) * (width_ - 2 * pad_w)]);
    }
  });
}

template <typename T, Layout L>
void crop(const Tensor<T, L> &input, Tensor<T, L> &result, const size_t start_h,
          const size_t start_w, const size_t end_h, const size_t end_w) {
  throw std::runtime_error("Unsupported tensor layout for cropping");
}

template <typename T>
void crop(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, const size_t start_h,
          const size_t start_w, const size_t end_h, const size_t end_w) {
  if (!input.is_on_cpu()) {
    throw std::runtime_error("crop requires CPU tensor");
  }

  if (end_h >= input.height() || end_w >= input.width() || start_h > end_h || start_w > end_w) {
    throw std::invalid_argument("Invalid crop dimensions");
  }

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height_ = input.height();
  const size_t width_ = input.width();

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < result.height(); ++h) {
        std::copy(&input_data[((n * channels + c) * height_ + (h + start_h)) * width_ + start_w],
                  &input_data[((n * channels + c) * height_ + (h + start_h)) * width_ + start_w] +
                      result.width(),
                  &result_data[((n * channels + c) * result.height() + h) * result.width()]);
      }
    }
  }
}

/**
 * @brief Slice the tensor along the batch dimension.
 * @param start_batch Starting batch index (inclusive)
 * @param end_batch Ending batch index (exclusive)
 * @return A new tensor containing the sliced batches
 */
template <typename T, Layout L>
void slice_batch(const Tensor<T, L> &input, Tensor<T, L> &result, size_t start_batch,
                 size_t end_batch) {
  if (!input.is_on_cpu()) {
    throw std::runtime_error("slice_batch requires CPU tensor");
  }

  if (end_batch > input.batch_size() || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  auto input_shape = input.shape();
  std::vector<size_t> result_shape = input_shape;
  result_shape[0] = end_batch - start_batch;
  result.resize(result_shape, input.device());

  const T *input_data = input.data_ptr().get();
  const std::vector<size_t> strides = input.strides();
  T *result_data = result.data_ptr().get();

  std::copy(&input_data[start_batch * strides[0]], &input_data[end_batch * strides[0]],
            result_data);
}

/*
 * @brief Slice the tensor along the channel dimension.
 */
template <typename T, Layout L>
void slice_channels(const Tensor<T, L> &input, Tensor<T, L> &result, size_t start_ch,
                    size_t end_ch) {
  throw std::runtime_error("Unsupported tensor layout for channel slicing");
}

/**
 * @brief Slice the tensor along the channel dimension.
 * @param start_ch Starting channel index (inclusive)
 * @param end_ch Ending channel index (inclusive)
 * @return A new tensor containing the sliced channels
 */
template <typename T>
void slice_channels(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t start_ch,
                    size_t end_ch) {
  if (end_ch >= input.channels() || start_ch > end_ch) {
    throw std::invalid_argument("Invalid channel slice range");
  }

  for (size_t n = 0; n < input.batch_size(); ++n) {
    for (size_t c = 0; c < result.channels(); ++c) {
      for (size_t h = 0; h < input.height(); ++h) {
        for (size_t w = 0; w < input.width(); ++w) {
          result(n, c, h, w) = input(n, start_ch + c, h, w);
        }
      }
    }
  }
}

template <typename T, Layout L>
void split(const Tensor<T, L> &input, std::vector<Tensor<T, L>> &results, size_t num_splits) {
  if (num_splits == 0 || num_splits > input.batch_size()) {
    throw std::invalid_argument("Invalid number of splits");
  }

  results.resize(num_splits);

  size_t split_size = input.batch_size() / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? input.batch_size() : start + split_size;

    cpu::slice_batch(input, results[i], start, end);
  }
}

template <typename T>
void split(const Tensor<T, NCHW> &input, std::vector<Tensor<T, NCHW>> &results, size_t num_splits) {
  if (num_splits == 0 || num_splits > input.batch_size()) {
    throw std::invalid_argument("Invalid number of splits");
  }

  size_t split_size = input.batch_size() / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? input.batch_size() : start + split_size;

    cpu::slice_batch(input, results[i], start, end);
  }
}

template <typename T, Layout L> void apply_softmax(Tensor<T, L> &input) {
  throw std::runtime_error("Unsupported tensor layout for softmax");
}

template <typename T> void apply_softmax(Tensor<T, NCHW> &input) {
  auto shape = input.shape();
  const size_t batch_size = shape[0];
  const size_t num_classes = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  // Apply softmax across channels at each spatial location
  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        // Find max value for numerical stability
        T max_val = input(batch, 0, h, w);
        for (size_t c = 1; c < num_classes; ++c) {
          max_val = std::max(max_val, input(batch, c, h, w));
        }

        // Apply exp and sum
        T sum = T(0);
        for (size_t c = 0; c < num_classes; ++c) {
          const T exp_val = std::exp(input(batch, c, h, w) - max_val);
          input(batch, c, h, w) = exp_val;
          sum += exp_val;
        }

        // Normalize with numerical stability protection
        const T inv_sum = T(1) / std::max(sum, static_cast<T>(1e-8));
        for (size_t c = 0; c < num_classes; ++c) {
          input(batch, c, h, w) *= inv_sum;
        }
      }
    }
  }
}

} // namespace cpu
} // namespace tnn