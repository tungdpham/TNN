#include "ops/cpu/dkernels.hpp"

#include "threading/thread_handler.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tnn {
namespace ops {
namespace cpu {
namespace dp {
#ifdef __AVX2__

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

inline void avx2_unaligned_sub_scalar(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_sub_pd(vec_a, vec_scalar);
    _mm256_storeu_pd(&c[i], vec_c);
  }
  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - scalar;
  }
}

inline void avx2_aligned_sub_scalar(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_scalar = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_sub_pd(vec_a, vec_scalar);
    _mm256_store_pd(&c[i], vec_c);
  }
  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - scalar;
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

inline void avx2_unaligned_scalar_max(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_b = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_c = _mm256_max_pd(vec_a, vec_b);
    _mm256_storeu_pd(&c[i], vec_c);
  }
  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], scalar);
  }
}

inline void avx2_aligned_scalar_max(const double *a, double scalar, double *c, size_t size) {
  __m256d vec_b = _mm256_set1_pd(scalar);
  size_t vec_size = (size / 4) * 4;
  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_c = _mm256_max_pd(vec_a, vec_b);
    _mm256_store_pd(&c[i], vec_c);
  }
  for (size_t i = vec_size; i < size; ++i) {
    c[i] = std::max(a[i], scalar);
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

inline double avx2_unaligned_sum(const double *a, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    sum = _mm256_add_pd(sum, vec_a);
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

inline void avx2_aligned_sum(const double *a, size_t size, double *result) {
  __m256d sum = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    sum = _mm256_add_pd(sum, vec_a);
  }

  __m128d hi = _mm256_extractf128_pd(sum, 1);
  __m128d lo = _mm256_castpd256_pd128(sum);
  __m128d sum128 = _mm_add_pd(hi, lo);
  sum128 = _mm_hadd_pd(sum128, sum128);
  double temp_result = _mm_cvtsd_f64(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    temp_result += a[i];
  }

  *result = temp_result;
}

inline double avx2_unaligned_dot_product(const double *a, const double *b, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d vec_b = _mm256_loadu_pd(&b[i]);
    sum = _mm256_fmadd_pd(vec_a, vec_b, sum);
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

inline double avx2_aligned_dot_product(const double *a, const double *b, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d vec_b = _mm256_load_pd(&b[i]);
    sum = _mm256_fmadd_pd(vec_a, vec_b, sum);
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

inline double avx2_unaligned_sum_squared_diff(const double *a, double mean, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  __m256d vec_mean = _mm256_set1_pd(mean);
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_loadu_pd(&a[i]);
    __m256d diff = _mm256_sub_pd(vec_a, vec_mean);
    sum = _mm256_fmadd_pd(diff, diff, sum);
  }

  __m128d hi = _mm256_extractf128_pd(sum, 1);
  __m128d lo = _mm256_castpd256_pd128(sum);
  __m128d sum128 = _mm_add_pd(hi, lo);
  sum128 = _mm_hadd_pd(sum128, sum128);
  double result = _mm_cvtsd_f64(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    double diff = a[i] - mean;
    result += diff * diff;
  }

  return result;
}

inline double avx2_aligned_sum_squared_diff(const double *a, double mean, size_t size) {
  __m256d sum = _mm256_setzero_pd();
  __m256d vec_mean = _mm256_set1_pd(mean);
  size_t vec_size = (size / 4) * 4;

  for (size_t i = 0; i < vec_size; i += 4) {
    __m256d vec_a = _mm256_load_pd(&a[i]);
    __m256d diff = _mm256_sub_pd(vec_a, vec_mean);
    sum = _mm256_fmadd_pd(diff, diff, sum);
  }

  __m128d hi = _mm256_extractf128_pd(sum, 1);
  __m128d lo = _mm256_castpd256_pd128(sum);
  __m128d sum128 = _mm_add_pd(hi, lo);
  sum128 = _mm_hadd_pd(sum128, sum128);
  double result = _mm_cvtsd_f64(sum128);

  for (size_t i = vec_size; i < size; ++i) {
    double diff = a[i] - mean;
    result += diff * diff;
  }

  return result;
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

void add(const double *a, const double *b, double *c, size_t size) {
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

void sub(const double *a, const double *b, double *c, size_t size) {
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

void mul(const double *a, const double *b, double *c, size_t size) {
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

void div(const double *a, const double *b, double *c, size_t size) {
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

void fmadd(const double *a, const double *b, double *c, size_t size) {
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

void fmsub(const double *a, const double *b, double *c, size_t size) {
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

void fnmadd(const double *a, const double *b, double *c, size_t size) {
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

void add_scalar(const double *a, double scalar, double *c, size_t size) {
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

void sub_scalar(const double *a, double scalar, double *c, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(c) % 32 == 0) {
    avx2_aligned_sub_scalar(a, scalar, c, size);
  } else {
    avx2_unaligned_sub_scalar(a, scalar, c, size);
  }
#else
  for (size_t i = 0; i < size; ++i) {
    c[i] = a[i] - scalar;
  }
#endif
}

void mul_scalar(const double *a, double scalar, double *c, size_t size) {
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

void div_scalar(const double *a, double scalar, double *c, size_t size) {
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

void set_scalar(double *c, double scalar, size_t size) {
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

// BLAS-like Operations
void axpy(double alpha, const double *x, double *y, size_t size) {
#ifdef __AVX2__
  const size_t simd_width = 4;
  const size_t simd_end = size - (size % simd_width);
  __m256d alpha_vec = _mm256_set1_pd(alpha);

  for (size_t i = 0; i < simd_end; i += simd_width) {
    __m256d x_vec = _mm256_loadu_pd(&x[i]);
    __m256d y_vec = _mm256_loadu_pd(&y[i]);
    y_vec = _mm256_fmadd_pd(alpha_vec, x_vec, y_vec);
    _mm256_storeu_pd(&y[i], y_vec);
  }

  for (size_t i = simd_end; i < size; ++i) {
    y[i] += alpha * x[i];
  }
#else
  for (size_t i = 0; i < size; ++i) {
    y[i] += alpha * x[i];
  }
#endif
}

void sqrt(const double *a, double *c, size_t size) {
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

void abs(const double *a, double *c, size_t size) {
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

void min(const double *a, const double *b, double *c, size_t size) {
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

void max(const double *a, const double *b, double *c, size_t size) {
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

void scalar_max(const double *a, double scalar, double *c, size_t size) {
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

void clamp(const double *a, double min_val, double max_val, double *c, size_t size) {
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

void equal(const double *a, const double *b, double *c, size_t size) {
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

void greater(const double *a, const double *b, double *c, size_t size) {
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

void copy(const double *a, double *c, size_t size) {
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

void zero(double *c, size_t size) {
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

void sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c, size_t size) {
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

void mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c, size_t size) {
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

double sum(const double *a, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0) {

    double result;
    avx2_aligned_sum(a, size, &result);
    return result;
  } else {
    return avx2_unaligned_sum(a, size);
  }
#else
  double result = 0.0;
  for (size_t i = 0; i < size; ++i) {
    result += a[i];
  }
  return result;
#endif
}

double dot_product(const double *a, const double *b, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0) {
    return avx2_aligned_dot_product(a, b, size);
  } else {
    return avx2_unaligned_dot_product(a, b, size);
  }
#else
  double result = 0.0;
  for (size_t i = 0; i < size; ++i) {
    result += a[i] * b[i];
  }
  return result;
#endif
}

double sum_squared_diff(const double *a, double mean, size_t size) {
#ifdef __AVX2__

  if (reinterpret_cast<uintptr_t>(a) % 32 == 0) {
    return avx2_aligned_sum_squared_diff(a, mean, size);
  } else {
    return avx2_unaligned_sum_squared_diff(a, mean, size);
  }
#else
  double result = 0.0;
  for (size_t i = 0; i < size; ++i) {
    double diff = a[i] - mean;
    result += diff * diff;
  }
  return result;
#endif
}

void fill_random_uniform(double *data, size_t size, double min_val, double max_val,
                         unsigned long long seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(min_val, max_val);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dist(gen);
  }
}

void fill_random_normal(double *data, size_t size, double mean, double stddev,
                        unsigned long long seed) {
  std::mt19937_64 gen(seed);
  std::normal_distribution<double> dist(mean, stddev);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dist(gen);
  }
}

} // namespace dp
} // namespace cpu
} // namespace ops
} // namespace tnn