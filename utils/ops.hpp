#include <memory>
#include <stdlib.h>

namespace utils {
template <typename T>
void transpose_2d(const T *src, T *dst, size_t rows, size_t cols) {
  // Use cache-friendly blocking for large matrices
  const size_t block_size = 64; // Tuned for typical L1 cache

  if (rows * cols < 1024) {
    // Simple transpose for small matrices
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        dst[j * rows + i] = src[i * cols + j];
      }
    }
  } else {
    // Blocked transpose for larger matrices
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
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
  }
}

template <typename T>
// Optimized SIMD dot product for contiguous memory access
T simd_dot_product_contiguous(const T *weights, const T *col_data,
                              size_t kernel_size) {
  T sum = T(0);

  // Use SIMD for float type only
  if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX2__)
    // AVX2 implementation - process 8 floats at once
    __m256 sum_vec = _mm256_setzero_ps();
    size_t simd_end = kernel_size - (kernel_size % 8);

    for (size_t ks = 0; ks < simd_end; ks += 8) {
      // Load 8 weights (contiguous)
      __m256 w_vec = _mm256_loadu_ps(&weights[ks]);

      // Load 8 col_data values (now contiguous!)
      __m256 c_vec = _mm256_loadu_ps(&col_data[ks]);

      // Fused multiply-add
      sum_vec = _mm256_fmadd_ps(w_vec, c_vec, sum_vec);
    }

    // Horizontal sum of the vector
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);

    // Sum the 4 elements in the 128-bit vector
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum = _mm_cvtss_f32(sum_128);

    // Handle remaining elements
    for (size_t ks = simd_end; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }

    _mm256_zeroupper(); // Clear upper bits for AVX2

#elif defined(__SSE2__)
    // SSE2 implementation - process 4 floats at once
    __m128 sum_vec = _mm_setzero_ps();
    size_t simd_end = kernel_size - (kernel_size % 4);

    for (size_t ks = 0; ks < simd_end; ks += 4) {
      // Load 4 weights (contiguous)
      __m128 w_vec = _mm_loadu_ps(&weights[ks]);

      // Load 4 col_data values (now contiguous!)
      __m128 c_vec = _mm_loadu_ps(&col_data[ks]);

      // Multiply and add
      __m128 prod = _mm_mul_ps(w_vec, c_vec);
      sum_vec = _mm_add_ps(sum_vec, prod);
    }

    // Horizontal sum of the vector
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum = _mm_cvtss_f32(sum_vec);

    // Handle remaining elements
    for (size_t ks = simd_end; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }

#else
    // Fallback scalar implementation
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }
#endif
  } else {
    // For non-float types, use scalar implementation
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      sum += weights[ks] * col_data[ks];
    }
  }

  return sum;
}
} // namespace utils