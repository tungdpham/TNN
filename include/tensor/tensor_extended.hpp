/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "matrix/matrix.hpp"
#include "tensor.hpp"

template <typename T>
Matrix<T> im2col_padded(const Tensor<T, NCHW> &input_tensor, size_t kernel_h, size_t kernel_w,
                        size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
  const size_t in_h = input_tensor.height();
  const size_t in_w = input_tensor.width();
  const size_t padded_h = in_h + 2 * pad_h;
  const size_t padded_w = in_w + 2 * pad_w;
  const size_t out_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t out_w = (padded_w - kernel_w) / stride_w + 1;
  const size_t channels = input_tensor.channels();
  const size_t batch_size = input_tensor.batch_size();

  size_t col_height = channels * kernel_h * kernel_w;
  size_t col_width = out_h * out_w;
  Matrix<T> col_matrix(col_height, batch_size * col_width);

  const T *input_data = input_tensor.data();
  T *col_data = col_matrix.data();

  utils::parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    const T *input_channel_ptr = input_data + (n * channels + c) * in_h * in_w;

    for (size_t kh = 0; kh < kernel_h; ++kh) {
      for (size_t kw = 0; kw < kernel_w; ++kw) {
        size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
        T *col_row_ptr = col_data + col_row_idx * (batch_size * col_width) + n * col_width;

        const size_t h_start = (pad_h > kh) ? ((pad_h - kh + stride_h - 1) / stride_h) : 0;
        const size_t h_end = std::min(out_h, (in_h + pad_h - kh) / stride_h);

        const size_t w_start = (pad_w > kw) ? ((pad_w - kw + stride_w - 1) / stride_w) : 0;
        const size_t w_end = std::min(out_w, (in_w + pad_w - kw) / stride_w);

        std::fill(col_row_ptr, col_row_ptr + h_start * out_w, T(0));
        col_row_ptr += h_start * out_w;

        for (size_t h = h_start; h < h_end; ++h) {
          const size_t h_in = h * stride_h + kh - pad_h;
          const T *input_row = input_channel_ptr + h_in * in_w;

          std::fill(col_row_ptr, col_row_ptr + w_start, T(0));
          col_row_ptr += w_start;

          if (stride_w == 1) {
            const size_t w_in_start = w_start * stride_w + kw - pad_w;
            std::copy(input_row + w_in_start, input_row + w_in_start + (w_end - w_start),
                      col_row_ptr);
            col_row_ptr += (w_end - w_start);
          } else {
            for (size_t w = w_start; w < w_end; ++w) {
              const size_t w_in = w * stride_w + kw - pad_w;
              *col_row_ptr++ = input_row[w_in];
            }
          }

          std::fill(col_row_ptr, col_row_ptr + (out_w - w_end), T(0));
          col_row_ptr += (out_w - w_end);
        }

        std::fill(col_row_ptr, col_row_ptr + (out_h - h_end) * out_w, T(0));
        col_row_ptr += (out_h - h_end) * out_w;
      }
    }
  });

  return col_matrix;
}

/**
 * @brief Convert a 4D image tensor to a column matrix for convolution.
 * @param kernel_h Height of the convolution kernel.
 * @param kernel_w Width of the convolution kernel.
 * @param stride_h Vertical stride of the convolution.
 * @param stride_w Horizontal stride of the convolution.
 * @param pad_h Vertical padding to be applied to the input tensor.
 * @param pad_w Horizontal padding to be applied to the input tensor.
 * @return A 2D matrix where each column corresponds to a flattened receptive field.
 */
template <typename T>
Matrix<T> im2col(const Tensor<T, NCHW> &input_tensor, size_t kernel_h, size_t kernel_w,
                 size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0) {
  if (pad_h > 0 || pad_w > 0) {
    return im2col_padded(input_tensor, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  }

  const size_t in_h = input_tensor.height();
  const size_t in_w = input_tensor.width();
  const size_t out_h = (in_h - kernel_h) / stride_h + 1;
  const size_t out_w = (in_w - kernel_w) / stride_w + 1;
  const size_t channels = input_tensor.channels();
  const size_t batch_size = input_tensor.batch_size();

  size_t col_height = channels * kernel_h * kernel_w;
  size_t col_width = out_h * out_w;
  Matrix<T> col_matrix(col_height, batch_size * col_width);

  const T *input_data = input_tensor.data();
  T *col_data = col_matrix.data();

  utils::parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
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
  });

  return col_matrix;
}

template <typename T>
static Tensor<T, NCHW> col2im_padded(const Matrix<T> &col_matrix, size_t batch_size,
                                     size_t channels, size_t height, size_t width, size_t kernel_h,
                                     size_t kernel_w, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w) {
  const size_t padded_h = height + 2 * pad_h;
  const size_t padded_w = width + 2 * pad_w;
  const size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  Tensor<T, NCHW> result(batch_size, channels, height, width);

  const T *col_data = col_matrix.data();
  T *result_data = result.data();
  const size_t col_width = output_h * output_w;

  utils::parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    T *result_channel_ptr = result_data + (n * channels + c) * height * width;

    for (size_t kh = 0; kh < kernel_h; ++kh) {
      for (size_t kw = 0; kw < kernel_w; ++kw) {
        size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
        const T *col_row_ptr = col_data + col_row_idx * (batch_size * col_width) + n * col_width;

        const size_t h_start = pad_h > kh ? ((pad_h - kh + stride_h - 1) / stride_h) : 0;
        const size_t h_end = std::min(output_h, (height + pad_h - kh) / stride_h);

        const size_t w_start = pad_w > kw ? ((pad_w - kw + stride_w - 1) / stride_w) : 0;
        const size_t w_end = std::min(output_w, (width + pad_w - kw) / stride_w);

        col_row_ptr += h_start * output_w;

        for (size_t h = h_start; h < h_end; ++h) {
          const size_t h_out = h * stride_h + kh - pad_h;
          T *result_row = result_channel_ptr + h_out * width;

          col_row_ptr += w_start;

          size_t valid_width = w_end - w_start;
          if (stride_w == 1) {
            size_t w_out_start = w_start * stride_w + kw - pad_w;
            utils::avx2_add(result_row + w_out_start, col_row_ptr, result_row + w_out_start,
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
  });

  return result;
}

/**
 * @brief Convert a column matrix back to the original image tensor.
 * @param col_matrix The input column matrix.
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
 * @return The reconstructed image tensor.
 */
template <typename T>
static Tensor<T, NCHW> col2im(const Matrix<T> &col_matrix, size_t batch_size, size_t channels,
                              size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                              size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
  if (pad_h > 0 || pad_w > 0) {
    return col2im_padded(col_matrix, batch_size, channels, height, width, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w);
  }
  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  Tensor<T, NCHW> result(batch_size, channels, padded_h, padded_w);

  const T *col_data = col_matrix.data();
  T *result_data = result.data();
  const size_t col_width = output_h * output_w;

  utils::parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    T *result_channel_ptr = result_data + (n * channels + c) * padded_h * padded_w;

    for (size_t kh = 0; kh < kernel_h; ++kh) {
      for (size_t kw = 0; kw < kernel_w; ++kw) {
        size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
        const T *col_row_ptr = col_data + col_row_idx * (batch_size * col_width) + n * col_width;

        for (size_t h = 0; h < output_h; ++h) {
          T *result_row_ptr = result_channel_ptr + (h * stride_h + kh) * padded_w + kw;

          if (stride_w == 1) {
            utils::avx2_add(result_row_ptr, col_row_ptr, result_row_ptr, output_w);
            col_row_ptr += output_w;
          } else {
            for (size_t w = 0; w < output_w; ++w) {
              result_row_ptr[w * stride_w] += *col_row_ptr++;
            }
          }
        }
      }
    }
  });

  return result;
}