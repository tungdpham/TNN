/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <stdlib.h>
#if defined(__AVX2__) || defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))
#include <immintrin.h>
#endif

#include "parallel_for.hpp"

namespace utils {

template <typename T>
void nchw_to_cnhw(const T *src, T *dst, size_t batch_size, size_t channels,
                  size_t height, size_t width) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      std::copy(&src[n * channels * height * width + c * height * width],
                &src[n * channels * height * width + c * height * width +
                     height * width],
                &dst[c * batch_size * height * width + n * height * width]);
    }
  }
#elif defined(USE_TBB)
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[n * channels * height * width + c * height * width],
              &src[n * channels * height * width + c * height * width +
                   height * width],
              &dst[c * batch_size * height * width + n * height * width]);
  });
#endif
}

template <typename T>
void cnhw_to_nchw(const T *src, T *dst, size_t batch_size, size_t channels,
                  size_t height, size_t width) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      std::copy(&src[c * batch_size * height * width + n * height * width],
                &src[c * batch_size * height * width + n * height * width +
                     height * width],
                &dst[n * channels * height * width + c * height * width]);
    }
  }
#elif defined(USE_TBB)
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[c * batch_size * height * width + n * height * width],
              &src[c * batch_size * height * width + n * height * width +
                   height * width],
              &dst[n * channels * height * width + c * height * width]);
  });
#endif
}

template <typename T>
void transpose_2d_inplace(const T *src, T *dst, size_t rows, size_t cols) {
#if defined(_OPENMP)
  const size_t block_size = 64;

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < rows; i += block_size) {
    for (size_t j = 0; j < cols; j += block_size) {
      size_t max_i = std::min(i + block_size, rows);
      size_t max_j = std::min(j + block_size, cols);

      for (size_t ii = i; ii < max_i; ++ii) {
        for (size_t jj = j; jj < max_j; ++jj) {
          dst[jj * rows + ii] = src[ii * cols + jj];
        }
      }
    }
  }
#elif defined(USE_TBB)
  parallel_for_2d(rows, cols, [&](size_t i, size_t j) {
    dst[j * rows + i] = src[i * cols + j];
  });
#endif
}

template <typename T> void apply_softmax(Tensor<float> &tensor) {
  const size_t batch_size = tensor.shape()[0];
  const size_t num_classes = tensor.shape()[1];

#ifdef _OPENMP
#pragma omp parallel for if (batch_size > 16)
#endif
  for (size_t batch = 0; batch < batch_size; ++batch) {
    float max_val = tensor(batch, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      max_val = std::max(max_val, tensor(batch, j, 0, 0));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      const float exp_val = std::exp(tensor(batch, j, 0, 0) - max_val);
      tensor(batch, j, 0, 0) = exp_val;
      sum += exp_val;
    }

    const float inv_sum = 1.0f / std::max(sum, 1e-8f);
    for (size_t j = 0; j < num_classes; ++j) {
      tensor(batch, j, 0, 0) *= inv_sum;
    }
  }
}

template <typename T>
float compute_class_accuracy(const Tensor<float> &predictions,
                             const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  int total_correct = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total_correct) if (batch_size > 16)
#endif
  for (size_t i = 0; i < batch_size; ++i) {

    int pred_class = 0;
    float max_pred = predictions(i, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      const float pred_val = predictions(i, j, 0, 0);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets(i, j, 0, 0) > 0.5f) {
        true_class = static_cast<int>(j);
        break;
      }
    }

    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return static_cast<float>(total_correct) / static_cast<float>(batch_size);
}

template <typename T>

T simd_dot_product(const T *weights, const T *col_data, size_t kernel_size) {
  T sum = T(0);

  if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(_M_X64))

    __m256 sum_vec = _mm256_setzero_ps();
    size_t simd_end = kernel_size - (kernel_size % 8);

    for (size_t ks = 0; ks < simd_end; ks += 8) {
      __m256 w_vec = _mm256_loadu_ps(&weights[ks]);

      __m256 c_vec = _mm256_loadu_ps(&col_data[ks]);

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
    std::cerr << "Warning: SIMD not supported, using scalar dot product."
              << std::endl;
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