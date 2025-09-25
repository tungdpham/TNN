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
#include <memory>
#include <stdlib.h>
#if defined(__AVX2__) || defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))
#include <immintrin.h>
#endif

#include "parallel_for.hpp"
#include "simd_asm.hpp"
#include "tensor/tensor.hpp"

namespace utils {

template <typename T>
void nchw_to_cnhw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[n * channels * height * width + c * height * width],
              &src[n * channels * height * width + c * height * width + height * width],
              &dst[c * batch_size * height * width + n * height * width]);
  });
}

template <typename T>
void cnhw_to_nchw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[c * batch_size * height * width + n * height * width],
              &src[c * batch_size * height * width + n * height * width + height * width],
              &dst[n * channels * height * width + c * height * width]);
  });
}

template <typename T>
void transpose_2d(const T *src, T *dst, const size_t rows, const size_t cols) {
#if defined(_OPENMP)
  constexpr size_t block_size = 64;
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
  parallel_for_2d(rows, cols, [&](size_t i, size_t j) { dst[j * rows + i] = src[i * cols + j]; });
#endif
}

template <typename T> void apply_softmax(Tensor<float> &tensor) {
  const size_t batch_size = tensor.shape()[0];
  const size_t num_classes = tensor.shape()[1];

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
float compute_class_accuracy(const Tensor<float> &predictions, const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  int total_correct = 0;

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

template <typename T> T simd_dot_product(const T *weights, const T *col_data, size_t kernel_size) {
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

// GEMM blocking parameters (tunable for different CPUs)
constexpr size_t GEMM_MC = 256; // M dimension blocking
constexpr size_t GEMM_KC = 128; // K dimension blocking
constexpr size_t GEMM_NC = 256; // N dimension blocking

// Helper for aligned memory allocation
template <typename T> static T *aligned_malloc_gemm(size_t count, size_t align = 64) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
  size_t size = count * sizeof(T);
  return static_cast<T *>(aligned_alloc(align, ((size + align - 1) / align) * align));
#else
  void *p = nullptr;
  size_t size = count * sizeof(T);
  int result = posix_memalign(&p, align, ((size + align - 1) / align) * align);
  return (result == 0) ? static_cast<T *>(p) : nullptr;
#endif
}

// Pack B matrix block for better cache locality
// Pack B[0..kc-1, 0..nc-1] into packed_B as column-major
template <typename T>
static void pack_b_block_gemm(const T *B_panel, T *packed_B, size_t ldb, size_t kc, size_t nc,
                              size_t jc) {
  for (size_t j = 0; j < nc; ++j) {
    T *dst_col = packed_B + j * kc;
    const T *src = B_panel + j; // Column j of B_panel (jc offset already in B_panel)
    for (size_t p = 0; p < kc; ++p) {
      dst_col[p] = src[p * ldb];
    }
  }
}

// GEMM micro-kernel with SIMD optimization for float
template <typename T>
static void gemm_panel_kernel(size_t mc, size_t nc, size_t kc, const T *A, size_t lda,
                              const T *packed_B, T *C, size_t ldc) {
  if constexpr (std::is_same_v<T, float>) {
    // Optimized version for float with SIMD
    for (size_t i = 0; i < mc; ++i) {
      const T *a_row = A + i * lda;
      T *c_row = C + i * ldc;
      for (size_t j = 0; j < nc; ++j) {
        const T *b_col = packed_B + j * kc;
        c_row[j] += simd_dot_product(a_row, b_col, kc);
      }
    }
  } else {
    // Generic version for other types
    for (size_t i = 0; i < mc; ++i) {
      const T *a_row = A + i * lda;
      T *c_row = C + i * ldc;
      for (size_t j = 0; j < nc; ++j) {
        const T *b_col = packed_B + j * kc;
        T acc = c_row[j];
        for (size_t p = 0; p < kc; ++p) {
          acc += a_row[p] * b_col[p];
        }
        c_row[j] = acc;
      }
    }
  }
}

template <typename T> void gemm(const T *a, const T *b, T *c, size_t M, size_t N, size_t K) {
  // C = A * B where A is M x K, B is K x N, C is M x N
  // All matrices are stored in row-major format
  const size_t lda = K; // Leading dimension of A
  const size_t ldb = N; // Leading dimension of B
  const size_t ldc = N; // Leading dimension of C

  // Allocate scratch buffer for packing B panels
  T *packed_B = aligned_malloc_gemm<T>(GEMM_KC * GEMM_NC);
  if (!packed_B) {
    // Fallback to simple implementation if allocation fails
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        T sum = T(0);
        for (size_t k = 0; k < K; ++k) {
          sum += a[i * lda + k] * b[k * ldb + j];
        }
        c[i * ldc + j] = sum;
      }
    }
    return;
  }

  // Blocked GEMM algorithm
  for (size_t jc = 0; jc < N; jc += GEMM_NC) {
    const size_t nc = std::min(GEMM_NC, N - jc);

    for (size_t pc = 0; pc < K; pc += GEMM_KC) {
      const size_t kc = std::min(GEMM_KC, K - pc);

      // Pack B panel B[pc:pc+kc-1, jc:jc+nc-1]
      const T *B_panel = b + pc * ldb + jc;
      pack_b_block_gemm(B_panel, packed_B, ldb, kc, nc, 0);

      // Parallelize over M dimension blocks
      parallel_for_range(size_t(0), (M + GEMM_MC - 1) / GEMM_MC, [&](size_t ic_idx) {
        const size_t ic = ic_idx * GEMM_MC;
        const size_t mc = std::min(GEMM_MC, M - ic);

        // A panel: A[ic:ic+mc-1, pc:pc+kc-1]
        const T *A_panel = a + ic * lda + pc;

        // C block: C[ic:ic+mc-1, jc:jc+nc-1]
        T *C_block = c + ic * ldc + jc;

        // Compute C_block += A_panel * packed_B
        gemm_panel_kernel(mc, nc, kc, A_panel, lda, packed_B, C_block, ldc);
      });
    }
  }

  std::free(packed_B);
}

} // namespace utils