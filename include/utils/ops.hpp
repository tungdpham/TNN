/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdlib.h>
#if defined(__AVX2__) || defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))
#include <immintrin.h>
#endif

#include "simd_asm.hpp"
#include "utils/avx2.hpp"

namespace utils {

template <typename T>
void nchw_to_cnhw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  tthreads::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[n * channels * height * width + c * height * width],
              &src[n * channels * height * width + c * height * width + height * width],
              &dst[c * batch_size * height * width + n * height * width]);
  });
}

template <typename T>
void cnhw_to_nchw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  tthreads::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[c * batch_size * height * width + n * height * width],
              &src[c * batch_size * height * width + n * height * width + height * width],
              &dst[n * channels * height * width + c * height * width]);
  });
}

/**
 * @brief Transpose a 2D matrix
 */
template <typename T>
void transpose_2d(const T *src, T *dst, const size_t rows, const size_t cols,
                  const size_t block_size = 64) {
  tthreads::parallel_for_2d((rows + block_size - 1) / block_size,
                            (cols + block_size - 1) / block_size,
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
inline T simd_dot_product_aligned(const T *weights, const T *col_data, size_t kernel_size) {
  T sum = T(0);

  if constexpr (std::is_same_v<T, float>) {
#if defined(__x86_64__) || defined(_M_X64)

    return simd_dot_product_asm_aligned(weights, col_data, kernel_size);
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(_M_X64))

    __m256 sum_vec = _mm256_setzero_ps();
    size_t simd_end = kernel_size ^ (kernel_size & 0x7);
    __m256 w_vec, c_vec;
    for (size_t ks = 0; ks < simd_end; ks += 8) {
      w_vec = _mm256_load_ps(&weights[ks]);

      c_vec = _mm256_load_ps(&col_data[ks]);

      sum_vec = _mm256_fmadd_ps(w_vec, c_vec, sum_vec);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum = _mm_cvtss_f32(sum_128);

    for (size_t ks = simd_end; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }

    _mm256_zeroupper();

#elif defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))

    __m128 sum_vec = _mm_setzero_ps();
    size_t simd_end = kernel_size - (kernel_size % 4);

    for (size_t ks = 0; ks < simd_end; ks += 4) {

      __m128 w_vec = _mm_load_ps(&weights[ks]);

      __m128 c_vec = _mm_load_ps(&col_data[ks]);

      __m128 prod = _mm_mul_ps(w_vec, c_vec);
      sum_vec = _mm_add_ps(sum_vec, prod);
    }

    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum = _mm_cvtss_f32(sum_vec);

    for (size_t ks = simd_end; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }

#else
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }
#endif
  } else {

    for (size_t ks = 0; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }
  }

  return sum;
}

template <typename T>
inline T simd_dot_product(const T *weights, const T *col_data, size_t kernel_size) {
  // delegate to aligned version if both pointers are aligned
  if ((((uintptr_t)weights) % 32 == 0) && (((uintptr_t)col_data) % 32 == 0)) {
    return simd_dot_product_aligned(weights, col_data, kernel_size);
  }
  T sum = T(0);

  if constexpr (std::is_same_v<T, float>) {
#if defined(__x86_64__) || defined(_M_X64)

    return simd_dot_product_asm(weights, col_data, kernel_size);
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(_M_X64))

    __m256 sum_vec = _mm256_setzero_ps();
    size_t simd_end = kernel_size ^ (kernel_size & 0x7);
    __m256 w_vec, c_vec;
    for (size_t ks = 0; ks < simd_end; ks += 8) {
      w_vec = _mm256_loadu_ps(&weights[ks]);

      c_vec = _mm256_loadu_ps(&col_data[ks]);

      sum_vec = _mm256_fmadd_ps(w_vec, c_vec, sum_vec);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum = _mm_cvtss_f32(sum_128);

    for (size_t ks = simd_end; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }

    _mm256_zeroupper();

#elif defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))

    __m128 sum_vec = _mm_setzero_ps();
    size_t simd_end = kernel_size - (kernel_size % 4);

    for (size_t ks = 0; ks < simd_end; ks += 4) {

      __m128 w_vec = _mm_loadu_ps(&weights[ks]);

      __m128 c_vec = _mm_loadu_ps(&col_data[ks]);

      __m128 prod = _mm_mul_ps(w_vec, c_vec);
      sum_vec = _mm_add_ps(sum_vec, prod);
    }

    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum = _mm_cvtss_f32(sum_vec);

    for (size_t ks = simd_end; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }

#else
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }
#endif
  } else {

    for (size_t ks = 0; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }
  }

  return sum;
}
} // namespace utils
