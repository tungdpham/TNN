#pragma once

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "threading/thread_handler.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace utils {

#ifdef __AVX2__

inline void avx2_unaligned_add(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_add_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

inline void avx2_aligned_add(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_add_ps(vec_a, vec_b);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

inline void avx2_unaligned_sub(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_sub_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

inline void avx2_aligned_sub(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_sub_ps(vec_a, vec_b);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

inline void avx2_unaligned_mul(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_mul_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

inline void avx2_aligned_mul(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_mul_ps(vec_a, vec_b);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

inline void avx2_unaligned_div(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_div_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}

inline void avx2_aligned_div(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_div_ps(vec_a, vec_b);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}

inline void avx2_unaligned_add(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_add_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

inline void avx2_aligned_add(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_add_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

inline void avx2_unaligned_sub(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_sub_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

inline void avx2_aligned_sub(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_sub_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

inline void avx2_unaligned_mul(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_mul_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

inline void avx2_aligned_mul(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_mul_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

inline void avx2_unaligned_div(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_div_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}

inline void avx2_aligned_div(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_div_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}

inline void avx2_unaligned_fmadd(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_loadu_pd(&c[i]);
    __m256d result = _mm256_fmadd_pd(vec_a, vec_b, vec_c);
    _mm256_storeu_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] + c[i];
  }
}

inline void avx2_aligned_fmadd(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_load_pd(&c[i]);
    __m256d result = _mm256_fmadd_pd(vec_a, vec_b, vec_c);
    _mm256_store_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] + c[i];
  }
}

inline void avx2_unaligned_fmsub(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_loadu_pd(&c[i]);
    __m256d result = _mm256_fmsub_pd(vec_a, vec_b, vec_c);
    _mm256_storeu_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] - c[i];
  }
}

inline void avx2_aligned_fmsub(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_load_pd(&c[i]);
    __m256d result = _mm256_fmsub_pd(vec_a, vec_b, vec_c);
    _mm256_store_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] - c[i];
  }
}

inline void avx2_unaligned_fnmadd(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_loadu_pd(&c[i]);
    __m256d result = _mm256_fnmadd_pd(vec_a, vec_b, vec_c);
    _mm256_storeu_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = -(a[i] * b[i]) + c[i];
  }
}

inline void avx2_aligned_fnmadd(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_load_pd(&c[i]);
    __m256d result = _mm256_fnmadd_pd(vec_a, vec_b, vec_c);
    _mm256_store_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = -(a[i] * b[i]) + c[i];
  }
}

inline void avx2_unaligned_add_scalar(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_add_pd(vec_a, vec_scalar);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + scalar;
  }
}

inline void avx2_aligned_add_scalar(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_add_pd(vec_a, vec_scalar);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + scalar;
  }
}

inline void avx2_unaligned_mul_scalar(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_mul_pd(vec_a, vec_scalar);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * scalar;
  }
}

inline void avx2_aligned_mul_scalar(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_mul_pd(vec_a, vec_scalar);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * scalar;
  }
}

inline void avx2_unaligned_div_scalar(const double *a, const double scalar, double *c,
                                      size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_div_pd(vec_a, vec_scalar);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / scalar;
  }
}

inline void avx2_aligned_div_scalar(const double *a, const double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_div_pd(vec_a, vec_scalar);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / scalar;
  }
}

inline void avx2_unaligned_set_scalar(double *c, double scalar, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    _mm256_storeu_pd(&c[i], vec_scalar);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = scalar;
  }
}

inline void avx2_aligned_set_scalar(double *c, double scalar, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    _mm256_store_pd(&c[i], vec_scalar);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = scalar;
  }
}

inline void avx2_unaligned_sqrt(const double *a, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_sqrt_pd(vec_a);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::sqrt(a[i]);
  }
}

inline void avx2_aligned_sqrt(const double *a, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_sqrt_pd(vec_a);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::sqrt(a[i]);
  }
}

inline void avx2_unaligned_abs(const double *a, double *c, size_t size) {
  __m256d sign_mask = _mm256_set1_pd(-0.0);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_andnot_pd(sign_mask, vec_a);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::abs(a[i]);
  }
}

inline void avx2_aligned_abs(const double *a, double *c, size_t size) {
  __m256d sign_mask = _mm256_set1_pd(-0.0);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_andnot_pd(sign_mask, vec_a);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::abs(a[i]);
  }
}

inline void avx2_unaligned_min(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_min_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
}

inline void avx2_aligned_min(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_min_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
}

inline void avx2_unaligned_max(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d vec_c = _mm256_max_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
}

inline void avx2_unaligned_scalar_max(const double *a, double b, double *c, size_t size) {
  __m256d vec_b = _mm256_set1_pd(b);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_max_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], b);
  }
}

inline void avx2_aligned_scalar_max(const double *a, double b, double *c, size_t size) {
  __m256d vec_b = _mm256_set1_pd(b);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_max_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], b);
  }
}

inline void avx2_aligned_max(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d vec_c = _mm256_max_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
}

inline void avx2_unaligned_clamp(const double *a, double min_val, double max_val, double *c,
                                 size_t size) {
  __m256d vec_min = _mm256_set1_pd(min_val);
  __m256d vec_max = _mm256_set1_pd(max_val);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_max_pd(_mm256_min_pd(vec_a, vec_max), vec_min);
    _mm256_storeu_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(min_val, std::min(max_val, a[i]));
  }
}

inline void avx2_aligned_clamp(const double *a, double min_val, double max_val, double *c,
                               size_t size) {
  __m256d vec_min = _mm256_set1_pd(min_val);
  __m256d vec_max = _mm256_set1_pd(max_val);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_max_pd(_mm256_min_pd(vec_a, vec_max), vec_min);
    _mm256_store_pd(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(min_val, std::min(max_val, a[i]));
  }
}

inline void avx2_unaligned_copy(const double *a, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    _mm256_storeu_pd(&c[i], vec_a);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i];
  }
}

inline void avx2_aligned_copy(const double *a, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    _mm256_store_pd(&c[i], vec_a);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i];
  }
}

inline void avx2_unaligned_zero(double *c, size_t size) {
  __m256d zero = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    _mm256_storeu_pd(&c[i], zero);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 0;
  }
}

inline void avx2_aligned_zero(double *c, size_t size) {
  __m256d zero = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    _mm256_store_pd(&c[i], zero);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 0;
  }
}

inline void avx2_unaligned_equal(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d mask = _mm256_cmp_pd(vec_a, vec_b, _CMP_EQ_OQ);
    __m256d result = _mm256_and_pd(mask, _mm256_set1_pd(1.0));
    _mm256_storeu_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] == b[i]) ? 1.0 : 0.0;
  }
}

inline void avx2_aligned_equal(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d mask = _mm256_cmp_pd(vec_a, vec_b, _CMP_EQ_OQ);
    __m256d result = _mm256_and_pd(mask, _mm256_set1_pd(1.0));
    _mm256_store_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] == b[i]) ? 1.0 : 0.0;
  }
}

inline void avx2_unaligned_greater(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    __m256d mask = _mm256_cmp_pd(vec_a, vec_b, _CMP_GT_OQ);
    __m256d result = _mm256_and_pd(mask, _mm256_set1_pd(1.0));
    _mm256_storeu_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] > b[i]) ? 1.0 : 0.0;
  }
}

inline void avx2_aligned_greater(const double *a, const double *b, double *c, size_t size) {
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    __m256d mask = _mm256_cmp_pd(vec_a, vec_b, _CMP_GT_OQ);
    __m256d result = _mm256_and_pd(mask, _mm256_set1_pd(1.0));
    _mm256_store_pd(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] > b[i]) ? 1.0 : 0.0;
  }
}

inline double avx2_sum(const double *a, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;

  if (reinterpret_cast<uintptr_t>(a) % 32 == 0) {
    for (size_t i = 0; i < vec_size; i += 4) {
      __m256d vec_a = _mm256_load_pd(&a[i]);
      sum = _mm256_add_pd(sum, vec_a);
    }
  } else {
    for (size_t i = 0; i < vec_size; i += 4) {
      __m256d vec_a = _mm256_loadu_pd(&a[i]);
      sum = _mm256_add_pd(sum, vec_a);
    }
  }

  __m128d hi = _mm256_extractf128_pd(sum, 1);
  __m128d lo = _mm256_castpd256_pd128(sum);
  __m128d sum128 = _mm_add_pd(hi, lo);
  sum128 = _mm_hadd_pd(sum128, sum128);
  double result = _mm_cvtsd_f64(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    result += a[i];
  }

  return result;
}

inline double avx2_dot_product(const double *a, const double *b, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;

  bool aligned =
      (reinterpret_cast<uintptr_t>(a) % 32 == 0) && (reinterpret_cast<uintptr_t>(b) % 32 == 0);

  if (aligned) {
    for (size_t i = 0; i < vec_size; i += 4) {
      __m256d vec_a = _mm256_load_pd(&a[i]);
      __m256d vec_b = _mm256_load_pd(&b[i]);
      sum = _mm256_fmadd_pd(vec_a, vec_b, sum);
    }
  } else {
    for (size_t i = 0; i < vec_size; i += 4) {
      __m256d vec_a = _mm256_loadu_pd(&a[i]);
      __m256d vec_b = _mm256_loadu_pd(&b[i]);
      sum = _mm256_fmadd_pd(vec_a, vec_b, sum);
    }
  }

  __m128d hi = _mm256_extractf128_pd(sum, 1);
  __m128d lo = _mm256_castpd256_pd128(sum);
  __m128d sum128 = _mm_add_pd(hi, lo);
  sum128 = _mm_hadd_pd(sum128, sum128);
  double result = _mm_cvtsd_f64(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

inline double avx2_norm_squared(const double *a, size_t size) {
  return avx2_dot_product(a, a, size);
}

inline void avx2_unaligned_fmadd(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_loadu_ps(&c[i]);
    __m256 result = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
    _mm256_storeu_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] + c[i];
  }
}

inline void avx2_aligned_fmadd(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_load_ps(&c[i]);
    __m256 result = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
    _mm256_store_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] + c[i];
  }
}

inline void avx2_unaligned_fmsub(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_loadu_ps(&c[i]);
    __m256 result = _mm256_fmsub_ps(vec_a, vec_b, vec_c);
    _mm256_storeu_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] - c[i];
  }
}

inline void avx2_aligned_fmsub(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_load_ps(&c[i]);
    __m256 result = _mm256_fmsub_ps(vec_a, vec_b, vec_c);
    _mm256_store_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * b[i] - c[i];
  }
}

inline void avx2_unaligned_fnmadd(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_loadu_ps(&c[i]);
    __m256 result = _mm256_fnmadd_ps(vec_a, vec_b, vec_c);
    _mm256_storeu_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = -(a[i] * b[i]) + c[i];
  }
}

inline void avx2_aligned_fnmadd(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_load_ps(&c[i]);
    __m256 result = _mm256_fnmadd_ps(vec_a, vec_b, vec_c);
    _mm256_store_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = -(a[i] * b[i]) + c[i];
  }
}

inline void avx2_unaligned_add_scalar(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_add_ps(vec_a, vec_scalar);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + scalar;
  }
}

inline void avx2_aligned_add_scalar(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_add_ps(vec_a, vec_scalar);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] + scalar;
  }
}

inline void avx2_unaligned_mul_scalar(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_mul_ps(vec_a, vec_scalar);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * scalar;
  }
}

inline void avx2_aligned_mul_scalar(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_mul_ps(vec_a, vec_scalar);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] * scalar;
  }
}

inline void avx2_unaligned_div_scalar(const float *a, const float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_div_ps(vec_a, vec_scalar);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / scalar;
  }
}

inline void avx2_aligned_div_scalar(const float *a, const float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_div_ps(vec_a, vec_scalar);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] / scalar;
  }
}

inline void avx2_unaligned_set_scalar(float *c, float scalar, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;

  tthreads::parallel_for<size_t>(0, vec_size / 8, [&](size_t block) {
    size_t i = block * 8;
    _mm256_storeu_ps(&c[i], vec_scalar);
  });

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = scalar;
  }
}

inline void avx2_aligned_set_scalar(float *c, float scalar, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  tthreads::parallel_for<size_t>(0, (size / 8), [&](size_t block) {
    size_t i = block * 8;
    _mm256_store_ps(&c[i], vec_scalar);
  });

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = scalar;
  }
}

inline void avx2_unaligned_sqrt(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_sqrt_ps(vec_a);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::sqrt(a[i]);
  }
}

inline void avx2_aligned_sqrt(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_sqrt_ps(vec_a);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::sqrt(a[i]);
  }
}

inline void avx2_unaligned_rsqrt(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_rsqrt_ps(vec_a);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 1.0f / std::sqrt(a[i]);
  }
}

inline void avx2_aligned_rsqrt(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_rsqrt_ps(vec_a);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 1.0f / std::sqrt(a[i]);
  }
}

inline void avx2_unaligned_rcp(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_rcp_ps(vec_a);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 1.0f / a[i];
  }
}

inline void avx2_aligned_rcp(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_rcp_ps(vec_a);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 1.0f / a[i];
  }
}

inline void avx2_unaligned_abs(const float *a, float *c, size_t size) {
  __m256 sign_mask = _mm256_set1_ps(-0.0f);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_andnot_ps(sign_mask, vec_a);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::abs(a[i]);
  }
}

inline void avx2_aligned_abs(const float *a, float *c, size_t size) {
  __m256 sign_mask = _mm256_set1_ps(-0.0f);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_andnot_ps(sign_mask, vec_a);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::abs(a[i]);
  }
}

inline void avx2_unaligned_min(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_min_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
}

inline void avx2_aligned_min(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_min_ps(vec_a, vec_b);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
}

inline void avx2_unaligned_max(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_max_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
}

inline void avx2_aligned_max(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 vec_c = _mm256_max_ps(vec_a, vec_b);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
}

inline void avx2_unaligned_scalar_max(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_max_ps(vec_a, vec_scalar);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], scalar);
  }
}

inline void avx2_aligned_scalar_max(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_max_ps(vec_a, vec_scalar);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], scalar);
  }
}

inline void avx2_unaligned_clamp(const float *a, float min_val, float max_val, float *c,
                                 size_t size) {
  __m256 vec_min = _mm256_set1_ps(min_val);
  __m256 vec_max = _mm256_set1_ps(max_val);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_max_ps(_mm256_min_ps(vec_a, vec_max), vec_min);
    _mm256_storeu_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(min_val, std::min(max_val, a[i]));
  }
}

inline void avx2_aligned_clamp(const float *a, float min_val, float max_val, float *c,
                               size_t size) {
  __m256 vec_min = _mm256_set1_ps(min_val);
  __m256 vec_max = _mm256_set1_ps(max_val);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_max_ps(_mm256_min_ps(vec_a, vec_max), vec_min);
    _mm256_store_ps(&c[i], vec_c);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(min_val, std::min(max_val, a[i]));
  }
}

inline void avx2_unaligned_copy(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;

  tthreads::parallel_for<size_t>(0, vec_size / 8, [&](size_t block) {
    size_t i = block * 8;
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    _mm256_storeu_ps(&c[i], vec_a);
  });

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i];
  }
}

inline void avx2_aligned_copy(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;

  tthreads::parallel_for<size_t>(0, vec_size / 8, [&](size_t block) {
    size_t i = block * 8;
    __m256 vec_a = _mm256_load_ps(&a[i]);
    _mm256_store_ps(&c[i], vec_a);
  });

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i];
  }
}

inline void avx2_unaligned_zero(float *c, size_t size) {
  __m256 zero = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    _mm256_storeu_ps(&c[i], zero);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 0;
  }
}

inline void avx2_aligned_zero(float *c, size_t size) {
  __m256 zero = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    _mm256_store_ps(&c[i], zero);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = 0;
  }
}

inline float avx2_sum(const float *a, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;

  if (reinterpret_cast<uintptr_t>(a) % 32 == 0) {
    for (size_t i = 0; i < vec_size; i += 8) {
      __m256 vec_a = _mm256_load_ps(&a[i]);
      sum = _mm256_add_ps(sum, vec_a);
    }
  } else {
    for (size_t i = 0; i < vec_size; i += 8) {
      __m256 vec_a = _mm256_loadu_ps(&a[i]);
      sum = _mm256_add_ps(sum, vec_a);
    }
  }

  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(hi, lo);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float result = _mm_cvtss_f32(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    result += a[i];
  }

  return result;
}

inline float avx2_dot_product(const float *a, const float *b, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;

  bool aligned =
      (reinterpret_cast<uintptr_t>(a) % 32 == 0) && (reinterpret_cast<uintptr_t>(b) % 32 == 0);

  if (aligned) {
    for (size_t i = 0; i < vec_size; i += 8) {
      __m256 vec_a = _mm256_load_ps(&a[i]);
      __m256 vec_b = _mm256_load_ps(&b[i]);
      sum = _mm256_fmadd_ps(vec_a, vec_b, sum);
    }
  } else {
    for (size_t i = 0; i < vec_size; i += 8) {
      __m256 vec_a = _mm256_loadu_ps(&a[i]);
      __m256 vec_b = _mm256_loadu_ps(&b[i]);
      sum = _mm256_fmadd_ps(vec_a, vec_b, sum);
    }
  }

  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(hi, lo);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float result = _mm_cvtss_f32(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

inline float avx2_norm_squared(const float *a, size_t size) { return avx2_dot_product(a, a, size); }

// Compute sum of squared differences: sum((a[i] - mean)^2)
inline float avx2_sum_squared_diff(const float *a, float mean, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  __m256 vec_mean = _mm256_set1_ps(mean);
  size_t vec_size = (size / 8) * 8;

  bool aligned = (reinterpret_cast<uintptr_t>(a) % 32 == 0);

  if (aligned) {
    for (size_t i = 0; i < vec_size; i += 8) {
      __m256 vec_a = _mm256_load_ps(&a[i]);
      __m256 diff = _mm256_sub_ps(vec_a, vec_mean);
      sum = _mm256_fmadd_ps(diff, diff, sum); // sum += diff * diff
    }
  } else {
    for (size_t i = 0; i < vec_size; i += 8) {
      __m256 vec_a = _mm256_loadu_ps(&a[i]);
      __m256 diff = _mm256_sub_ps(vec_a, vec_mean);
      sum = _mm256_fmadd_ps(diff, diff, sum); // sum += diff * diff
    }
  }

  // Horizontal sum of the vector
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(hi, lo);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float result = _mm_cvtss_f32(sum128);

  // Handle remaining elements
  for (size_t i = vec_size; i < size; ++i) {
    float diff = a[i] - mean;
    result += diff * diff;
  }

  return result;
}

// Double precision version
inline double avx2_sum_squared_diff(const double *a, double mean, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  __m256d vec_mean = _mm256_set1_pd(mean);
  size_t vec_size = (size / 4) * 4;

  bool aligned = (reinterpret_cast<uintptr_t>(a) % 32 == 0);

  if (aligned) {
    for (size_t i = 0; i < vec_size; i += 4) {
      __m256d vec_a = _mm256_load_pd(&a[i]);
      __m256d diff = _mm256_sub_pd(vec_a, vec_mean);
      sum = _mm256_fmadd_pd(diff, diff, sum); // sum += diff * diff
    }
  } else {
    for (size_t i = 0; i < vec_size; i += 4) {
      __m256d vec_a = _mm256_loadu_pd(&a[i]);
      __m256d diff = _mm256_sub_pd(vec_a, vec_mean);
      sum = _mm256_fmadd_pd(diff, diff, sum); // sum += diff * diff
    }
  }

  // Horizontal sum of the vector
  __m128d hi = _mm256_extractf128_pd(sum, 1);
  __m128d lo = _mm256_castpd256_pd128(sum);
  __m128d sum128 = _mm_add_pd(hi, lo);
  sum128 = _mm_hadd_pd(sum128, sum128);
  double result = _mm_cvtsd_f64(sum128);

  // Handle remaining elements
  for (size_t i = vec_size; i < size; ++i) {
    double diff = a[i] - mean;
    result += diff * diff;
  }

  return result;
}

inline void avx2_unaligned_equal(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 mask = _mm256_cmp_ps(vec_a, vec_b, _CMP_EQ_OQ);
    __m256 result = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
    _mm256_storeu_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
  }
}

inline void avx2_aligned_equal(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 mask = _mm256_cmp_ps(vec_a, vec_b, _CMP_EQ_OQ);
    __m256 result = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
    _mm256_store_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
  }
}

inline void avx2_unaligned_greater(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 mask = _mm256_cmp_ps(vec_a, vec_b, _CMP_GT_OQ);
    __m256 result = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
    _mm256_storeu_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
  }
}

inline void avx2_aligned_greater(const float *a, const float *b, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    __m256 mask = _mm256_cmp_ps(vec_a, vec_b, _CMP_GT_OQ);
    __m256 result = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
    _mm256_store_ps(&c[i], result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
  }
}

// Specialized operations for BatchNorm: (a - scalar1) * scalar2
inline void avx2_unaligned_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar,
                                          float *c, size_t size) {
  __m256 vec_sub = _mm256_set1_ps(sub_scalar);
  __m256 vec_mul = _mm256_set1_ps(mul_scalar);
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_temp = _mm256_sub_ps(vec_a, vec_sub);
    __m256 vec_result = _mm256_mul_ps(vec_temp, vec_mul);
    _mm256_storeu_ps(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] - sub_scalar) * mul_scalar;
  }
}

inline void avx2_aligned_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar,
                                        float *c, size_t size) {
  __m256 vec_sub = _mm256_set1_ps(sub_scalar);
  __m256 vec_mul = _mm256_set1_ps(mul_scalar);
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_temp = _mm256_sub_ps(vec_a, vec_sub);
    __m256 vec_result = _mm256_mul_ps(vec_temp, vec_mul);
    _mm256_store_ps(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] - sub_scalar) * mul_scalar;
  }
}

inline void avx2_unaligned_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar,
                                          double *c, size_t size) {
  __m256d vec_sub = _mm256_set1_pd(sub_scalar);
  __m256d vec_mul = _mm256_set1_pd(mul_scalar);
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_temp = _mm256_sub_pd(vec_a, vec_sub);
    __m256d vec_result = _mm256_mul_pd(vec_temp, vec_mul);
    _mm256_storeu_pd(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] - sub_scalar) * mul_scalar;
  }
}

inline void avx2_aligned_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar,
                                        double *c, size_t size) {
  __m256d vec_sub = _mm256_set1_pd(sub_scalar);
  __m256d vec_mul = _mm256_set1_pd(mul_scalar);
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_temp = _mm256_sub_pd(vec_a, vec_sub);
    __m256d vec_result = _mm256_mul_pd(vec_temp, vec_mul);
    _mm256_store_pd(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = (a[i] - sub_scalar) * mul_scalar;
  }
}

// Specialized operations for BatchNorm: scalar1 * a + scalar2
inline void avx2_unaligned_mul_add_scalar(const float *a, float mul_scalar, float add_scalar,
                                          float *c, size_t size) {
  __m256 vec_mul = _mm256_set1_ps(mul_scalar);
  __m256 vec_add = _mm256_set1_ps(add_scalar);
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_result = _mm256_fmadd_ps(vec_a, vec_mul, vec_add);
    _mm256_storeu_ps(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = mul_scalar * a[i] + add_scalar;
  }
}

inline void avx2_aligned_mul_add_scalar(const float *a, float mul_scalar, float add_scalar,
                                        float *c, size_t size) {
  __m256 vec_mul = _mm256_set1_ps(mul_scalar);
  __m256 vec_add = _mm256_set1_ps(add_scalar);
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_result = _mm256_fmadd_ps(vec_a, vec_mul, vec_add);
    _mm256_store_ps(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = mul_scalar * a[i] + add_scalar;
  }
}

inline void avx2_unaligned_mul_add_scalar(const double *a, double mul_scalar, double add_scalar,
                                          double *c, size_t size) {
  __m256d vec_mul = _mm256_set1_pd(mul_scalar);
  __m256d vec_add = _mm256_set1_pd(add_scalar);
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_result = _mm256_fmadd_pd(vec_a, vec_mul, vec_add);
    _mm256_storeu_pd(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = mul_scalar * a[i] + add_scalar;
  }
}

inline void avx2_aligned_mul_add_scalar(const double *a, double mul_scalar, double add_scalar,
                                        double *c, size_t size) {
  __m256d vec_mul = _mm256_set1_pd(mul_scalar);
  __m256d vec_add = _mm256_set1_pd(add_scalar);
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_result = _mm256_fmadd_pd(vec_a, vec_mul, vec_add);
    _mm256_store_pd(&c[i], vec_result);
  }

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = mul_scalar * a[i] + add_scalar;
  }
}
#endif

inline void avx2_add(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_add(a, b, c, size);
  } else {
    avx2_unaligned_add(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
#endif
}
inline void avx2_sub(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sub(a, b, c, size);
  } else {
    avx2_unaligned_sub(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
#endif
}

inline void avx2_mul(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_mul(a, b, c, size);
  } else {
    avx2_unaligned_mul(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
#endif
}

inline void avx2_div(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_div(a, b, c, size);
  } else {
    avx2_unaligned_div(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
#endif
}

inline void avx2_fmadd(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_fmadd(a, b, c, size);
  } else {
    avx2_unaligned_fmadd(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * b[i] + c[i];
  }
#endif
}

inline void avx2_fmsub(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_fmsub(a, b, c, size);
  } else {
    avx2_unaligned_fmsub(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * b[i] - c[i];
  }
#endif
}

inline void avx2_fnmadd(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_fnmadd(a, b, c, size);
  } else {
    avx2_unaligned_fnmadd(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = -(a[i] * b[i]) + c[i];
  }
#endif
}

inline void avx2_add(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_add(a, b, c, size);
  } else {
    avx2_unaligned_add(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
#endif
}

inline void avx2_sub(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sub(a, b, c, size);
  } else {
    avx2_unaligned_sub(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
#endif
}

inline void avx2_mul(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_mul(a, b, c, size);
  } else {
    avx2_unaligned_mul(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
#endif
}

inline void avx2_div(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_div(a, b, c, size);
  } else {
    avx2_unaligned_div(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
#endif
}

inline void avx2_fmadd(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_fmadd(a, b, c, size);
  } else {
    avx2_unaligned_fmadd(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * b[i] + c[i];
  }
#endif
}

inline void avx2_fmsub(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_fmsub(a, b, c, size);
  } else {
    avx2_unaligned_fmsub(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * b[i] - c[i];
  }
#endif
}

inline void avx2_fnmadd(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_fnmadd(a, b, c, size);
  } else {
    avx2_unaligned_fnmadd(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = -(a[i] * b[i]) + c[i];
  }
#endif
}

inline void avx2_add_scalar(const double *a, double scalar, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_add_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_add_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] + scalar;
  }
#endif
}

inline void avx2_mul_scalar(const double *a, double scalar, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_mul_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_mul_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * scalar;
  }
#endif
}

inline void avx2_div_scalar(const double *a, double scalar, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_div_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_div_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] / scalar;
  }
#endif
}

inline void avx2_set_scalar(double *c, double scalar, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_set_scalar(c, scalar, size);
  } else {
    avx2_unaligned_set_scalar(c, scalar, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = scalar;
  }
#endif
}

inline void avx2_sqrt(const double *a, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sqrt(a, c, size);
  } else {
    avx2_unaligned_sqrt(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::sqrt(a[i]);
  }
#endif
}

inline void avx2_abs(const double *a, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_abs(a, c, size);
  } else {
    avx2_unaligned_abs(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::abs(a[i]);
  }
#endif
}

inline void avx2_min(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_min(a, b, c, size);
  } else {
    avx2_unaligned_min(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
#endif
}

inline void avx2_max(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_max(a, b, c, size);
  } else {
    avx2_unaligned_max(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
#endif
}

inline void avx2_scalar_max(const double *a, double scalar, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_scalar_max(a, scalar, c, size);
  } else {
    avx2_unaligned_scalar_max(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::max(scalar, a[i]);
  }
#endif
}

inline void avx2_clamp(const double *a, double min_val, double max_val, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_clamp(a, min_val, max_val, c, size);
  } else {
    avx2_unaligned_clamp(a, min_val, max_val, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::max(min_val, std::min(max_val, a[i]));
  }
#endif
}

inline void avx2_equal(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_equal(a, b, c, size);
  } else {
    avx2_unaligned_equal(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = (a[i] == b[i]) ? 1.0 : 0.0;
  }
#endif
}

inline void avx2_greater(const double *a, const double *b, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_greater(a, b, c, size);
  } else {
    avx2_unaligned_greater(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = (a[i] > b[i]) ? 1.0 : 0.0;
  }
#endif
}

inline void avx2_copy(const double *a, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_copy(a, c, size);
  } else {
    avx2_unaligned_copy(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i];
  }
#endif
}

inline void avx2_zero(double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_zero(c, size);
  } else {
    avx2_unaligned_zero(c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = 0;
  }
#endif
}

inline void avx2_add_scalar(const float *a, float scalar, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_add_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_add_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] + scalar;
  }
#endif
}

inline void avx2_mul_scalar(const float *a, float scalar, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_mul_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_mul_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] * scalar;
  }
#endif
}

inline void avx2_div_scalar(const float *a, float scalar, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_div_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_div_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] / scalar;
  }
#endif
}

inline void avx2_set_scalar(float *c, float scalar, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_set_scalar(c, scalar, size);
  } else {
    avx2_unaligned_set_scalar(c, scalar, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = scalar;
  }
#endif
}

inline void avx2_sqrt(const float *a, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sqrt(a, c, size);
  } else {
    avx2_unaligned_sqrt(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::sqrt(a[i]);
  }
#endif
}

inline void avx2_rsqrt(const float *a, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_rsqrt(a, c, size);
  } else {
    avx2_unaligned_rsqrt(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = 1.0f / std::sqrt(a[i]);
  }
#endif
}

inline void avx2_rcp(const float *a, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_rcp(a, c, size);
  } else {
    avx2_unaligned_rcp(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = 1.0f / a[i];
  }
#endif
}

inline void avx2_abs(const float *a, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_abs(a, c, size);
  } else {
    avx2_unaligned_abs(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::abs(a[i]);
  }
#endif
}

inline void avx2_min(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_min(a, b, c, size);
  } else {
    avx2_unaligned_min(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
#endif
}

inline void avx2_max(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_max(a, b, c, size);
  } else {
    avx2_unaligned_max(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
#endif
}

inline void avx2_scalar_max(const float *a, float scalar, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_scalar_max(a, scalar, c, size);
  } else {
    avx2_unaligned_scalar_max(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::max(scalar, a[i]);
  }
#endif
}

inline void avx2_clamp(const float *a, float min_val, float max_val, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_clamp(a, min_val, max_val, c, size);
  } else {
    avx2_unaligned_clamp(a, min_val, max_val, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = std::max(min_val, std::min(max_val, a[i]));
  }
#endif
}

inline void avx2_equal(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_equal(a, b, c, size);
  } else {
    avx2_unaligned_equal(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
  }
#endif
}

inline void avx2_greater(const float *a, const float *b, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_greater(a, b, c, size);
  } else {
    avx2_unaligned_greater(a, b, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
  }
#endif
}

inline void avx2_copy(const float *a, float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_copy(a, c, size);
  } else {
    avx2_unaligned_copy(a, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i];
  }
#endif
}

inline void avx2_zero(float *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_zero(c, size);
  } else {
    avx2_unaligned_zero(c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = 0;
  }
#endif
}

// Public wrapper functions for BatchNorm optimizations
inline void avx2_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c,
                                size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
  } else {
    avx2_unaligned_sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = (a[i] - sub_scalar) * mul_scalar;
  }
#endif
}

inline void avx2_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c,
                                size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
  } else {
    avx2_unaligned_sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = (a[i] - sub_scalar) * mul_scalar;
  }
#endif
}

inline void avx2_mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c,
                                size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_mul_add_scalar(a, mul_scalar, add_scalar, c, size);
  } else {
    avx2_unaligned_mul_add_scalar(a, mul_scalar, add_scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = mul_scalar * a[i] + add_scalar;
  }
#endif
}

inline void avx2_mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c,
                                size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_mul_add_scalar(a, mul_scalar, add_scalar, c, size);
  } else {
    avx2_unaligned_mul_add_scalar(a, mul_scalar, add_scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = mul_scalar * a[i] + add_scalar;
  }
#endif
}

} // namespace utils