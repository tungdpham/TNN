#include "ops/cpu/skernels.hpp"

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
namespace fp {
#ifdef __AVX2__

// AVX2 Single-Precision Implementations (Float)

// Basic Arithmetic (Float)
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

// Fused Multiply-Add Operations (Float)
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

// Scalar Operations (Float)
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

inline void avx2_unaligned_sub_scalar(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_c = _mm256_sub_ps(vec_a, vec_scalar);
    _mm256_storeu_ps(&c[i], vec_c);
  }
  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - scalar;
  }
}

inline void avx2_aligned_sub_scalar(const float *a, float scalar, float *c, size_t size) {
  __m256 vec_scalar = _mm256_set1_ps(scalar);
  size_t vec_size = (size / 8) * 8;
  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_c = _mm256_sub_ps(vec_a, vec_scalar);
    _mm256_store_ps(&c[i], vec_c);
  }
  for (size_t i = vec_size; i < size; ++i) {
    c[i] = a[i] - scalar;
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

  parallel_for<size_t>(0, vec_size / 8, [&](size_t block) {
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
  parallel_for<size_t>(0, (size / 8), [&](size_t block) {
    size_t i = block * 8;
    _mm256_store_ps(&c[i], vec_scalar);
  });

  for (size_t i = vec_size; i < size; ++i) {
    c[i] = scalar;
  }
}

// Element-wise Functions (Float)
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

// Comparison and Clamping (Float)
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

// Memory and Reduction (Float)
inline void avx2_unaligned_copy(const float *a, float *c, size_t size) {
  size_t vec_size = (size / 8) * 8;

  parallel_for<size_t>(0, vec_size / 8, [&](size_t block) {
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

  parallel_for<size_t>(0, vec_size / 8, [&](size_t block) {
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

inline float avx2_unaligned_sum(const float *a, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    sum = _mm256_add_ps(sum, vec_a);
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

inline float avx2_aligned_sum(const float *a, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    sum = _mm256_add_ps(sum, vec_a);
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

inline float avx2_unaligned_dot_product(const float *a, const float *b, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    sum = _mm256_fmadd_ps(vec_a, vec_b, sum);
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

inline float avx2_aligned_dot_product(const float *a, const float *b, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 vec_b = _mm256_load_ps(&b[i]);
    sum = _mm256_fmadd_ps(vec_a, vec_b, sum);
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

inline float avx2_unaligned_sum_squared_diff(const float *a, float mean, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  __m256 vec_mean = _mm256_set1_ps(mean);
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 diff = _mm256_sub_ps(vec_a, vec_mean);
    sum = _mm256_fmadd_ps(diff, diff, sum); // sum += diff * diff
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

inline float avx2_aligned_sum_squared_diff(const float *a, float mean, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  __m256 vec_mean = _mm256_set1_ps(mean);
  size_t vec_size = (size / 8) * 8;

  for (size_t i = 0; i < vec_size; i += 8) {
    __m256 vec_a = _mm256_load_ps(&a[i]);
    __m256 diff = _mm256_sub_ps(vec_a, vec_mean);
    sum = _mm256_fmadd_ps(diff, diff, sum); // sum += diff * diff
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

// Specialized BatchNorm Operations (Float)
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

#endif // __AVX2__

// --- Wrapper Implementations (Float) ---
// These provide the dispatch logic to AVX2/Scalar

void add(const float *a, const float *b, float *c, size_t size) {
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

void sub(const float *a, const float *b, float *c, size_t size) {
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

void mul(const float *a, const float *b, float *c, size_t size) {
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

void div(const float *a, const float *b, float *c, size_t size) {
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

void fmadd(const float *a, const float *b, float *c, size_t size) {
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

void fmsub(const float *a, const float *b, float *c, size_t size) {
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

void fnmadd(const float *a, const float *b, float *c, size_t size) {
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

void add_scalar(const float *a, float scalar, float *c, size_t size) {
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

void sub_scalar(const float *a, float scalar, float *c, size_t size) {
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

void mul_scalar(const float *a, float scalar, float *c, size_t size) {
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

void div_scalar(const float *a, float scalar, float *c, size_t size) {
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

void set_scalar(float *c, float scalar, size_t size) {
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
void axpy(float alpha, const float *x, float *y, size_t size) {
#ifdef __AVX2__
  const size_t simd_width = 8;
  const size_t simd_end = size - (size % simd_width);
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  for (size_t i = 0; i < simd_end; i += simd_width) {
    __m256 x_vec = _mm256_loadu_ps(&x[i]);
    __m256 y_vec = _mm256_loadu_ps(&y[i]);
    y_vec = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);
    _mm256_storeu_ps(&y[i], y_vec);
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

void sqrt(const float *a, float *c, size_t size) {
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

void rsqrt(const float *a, float *c, size_t size) {
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

void rcp(const float *a, float *c, size_t size) {
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

void abs(const float *a, float *c, size_t size) {
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

void min(const float *a, const float *b, float *c, size_t size) {
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

void max(const float *a, const float *b, float *c, size_t size) {
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

void scalar_max(const float *a, float scalar, float *c, size_t size) {
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

void clamp(const float *a, float min_val, float max_val, float *c, size_t size) {
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

void equal(const float *a, const float *b, float *c, size_t size) {
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

void greater(const float *a, const float *b, float *c, size_t size) {
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

void copy(const float *a, float *c, size_t size) {
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

void zero(float *c, size_t size) {
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

// Specialized BatchNorm Operations (Float)
void sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c, size_t size) {
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

void mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c, size_t size) {
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

// Reduction Functions (Float)
float sum(const float *a, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0) {
    return avx2_aligned_sum(a, size);
  } else {
    return avx2_unaligned_sum(a, size);
  }
#else
  float result = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    result += a[i];
  }
  return result;
#endif
}

float dot_product(const float *a, const float *b, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0 && reinterpret_cast<uintptr_t>(b) % 32 == 0) {
    return avx2_aligned_dot_product(a, b, size);
  } else {
    return avx2_unaligned_dot_product(a, b, size);
  }
#else
  float result = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    result += a[i] * b[i];
  }
  return result;
#endif
}

float sum_squared_diff(const float *a, float mean, size_t size) {
#ifdef __AVX2__
  if (reinterpret_cast<uintptr_t>(a) % 32 == 0) {
    return avx2_aligned_sum_squared_diff(a, mean, size);
  } else {
    return avx2_unaligned_sum_squared_diff(a, mean, size);
  }
#else
  float result = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    float diff = a[i] - mean;
    result += diff * diff;
  }
  return result;
#endif
}

#ifdef __AVX2__
// Fast AVX2 approximations for transcendental functions

// Fast log approximation using polynomial (relative error ~0.2%)
inline __m256 _mm256_log_ps(__m256 x) {
  // Constants for log approximation
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(0.693147181f); // ln(2)
  const __m256 c2 = _mm256_set1_ps(-0.5f);
  const __m256 c3 = _mm256_set1_ps(0.333333333f);
  const __m256 c4 = _mm256_set1_ps(-0.25f);
  const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
  const __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
  const __m256i half = _mm256_set1_epi32(0x3F000000);

  // Extract exponent
  __m256i xi = _mm256_castps_si256(x);
  __m256i exp = _mm256_and_si256(xi, exp_mask);
  exp = _mm256_srli_epi32(exp, 23);
  exp = _mm256_sub_epi32(exp, _mm256_set1_epi32(127));
  __m256 e = _mm256_cvtepi32_ps(exp);

  // Extract mantissa [0.5, 1.0)
  __m256i mant = _mm256_and_si256(xi, mantissa_mask);
  mant = _mm256_or_si256(mant, half);
  __m256 m = _mm256_castsi256_ps(mant);

  // Polynomial approximation: log(m) where m in [0.5, 1.0)
  __m256 z = _mm256_sub_ps(m, one);
  __m256 z2 = _mm256_mul_ps(z, z);
  __m256 z3 = _mm256_mul_ps(z2, z);
  __m256 z4 = _mm256_mul_ps(z3, z);

  __m256 result = z;
  result = _mm256_fmadd_ps(z2, c2, result);
  result = _mm256_fmadd_ps(z3, c3, result);
  result = _mm256_fmadd_ps(z4, c4, result);
  result = _mm256_fmadd_ps(e, c1, result);

  return result;
}

// Fast cos approximation using minimax polynomial (error ~1e-5)
inline __m256 _mm256_cos_ps(__m256 x) {
  const __m256 two_pi = _mm256_set1_ps(6.28318530718f);
  const __m256 c1 = _mm256_set1_ps(-0.5f);
  const __m256 c2 = _mm256_set1_ps(0.04166666667f);
  const __m256 c3 = _mm256_set1_ps(-0.00138888889f);
  const __m256 c4 = _mm256_set1_ps(0.00002480159f);
  const __m256 one = _mm256_set1_ps(1.0f);

  // Range reduction: x = x mod 2π
  __m256 k = _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
  x = _mm256_fnmadd_ps(k, two_pi, x);

  // Use cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8!
  __m256 x2 = _mm256_mul_ps(x, x);
  __m256 x4 = _mm256_mul_ps(x2, x2);
  __m256 x6 = _mm256_mul_ps(x4, x2);
  __m256 x8 = _mm256_mul_ps(x4, x4);

  __m256 result = one;
  result = _mm256_fmadd_ps(x2, c1, result);
  result = _mm256_fmadd_ps(x4, c2, result);
  result = _mm256_fmadd_ps(x6, c3, result);
  result = _mm256_fmadd_ps(x8, c4, result);

  return result;
}
#endif

// Utility Functions (Float)
void fill_random_uniform(float *data, size_t size, float min_val, float max_val,
                         unsigned long long seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dist(gen);
  }
}

void fill_random_normal(float *data, size_t size, float mean, float stddev,
                        unsigned long long seed) {
#ifdef __AVX2__
  // Use vectorized Box-Muller for performance (10-20x faster than std::mt19937)
  size_t vec_size = (size / 8) * 8;
  const __m256 v_mean = _mm256_set1_ps(mean);
  const __m256 v_std = _mm256_set1_ps(stddev);
  const __m256 v_neg2 = _mm256_set1_ps(-2.0f);
  const __m256 v_2pi = _mm256_set1_ps(6.28318530718f);

  uint64_t rng_state = seed;

  for (size_t i = 0; i < vec_size; i += 8) {
    // Generate uniform random numbers using fast xorshift
    float u1[8], u2[8];
    for (int j = 0; j < 8; j++) {
      // Xorshift64 PRNG (very fast)
      rng_state ^= rng_state << 13;
      rng_state ^= rng_state >> 7;
      rng_state ^= rng_state << 17;
      u1[j] = (rng_state & 0xFFFFFF) / float(1 << 24);

      rng_state ^= rng_state << 13;
      rng_state ^= rng_state >> 7;
      rng_state ^= rng_state << 17;
      u2[j] = (rng_state & 0xFFFFFF) / float(1 << 24);
    }

    // Load uniform values
    __m256 U1 = _mm256_loadu_ps(u1);
    __m256 U2 = _mm256_loadu_ps(u2);

    // Box-Muller transform: z = sqrt(-2*ln(u1)) * cos(2π*u2)
    __m256 r = _mm256_sqrt_ps(_mm256_mul_ps(v_neg2, _mm256_log_ps(U1)));
    __m256 theta = _mm256_mul_ps(U2, v_2pi);
    __m256 z = _mm256_mul_ps(r, _mm256_cos_ps(theta));

    // Scale and shift: output = mean + stddev * z
    __m256 out = _mm256_fmadd_ps(z, v_std, v_mean);

    // Store result (use aligned store if buffer is aligned)
    if (reinterpret_cast<uintptr_t>(&data[i]) % 32 == 0) {
      _mm256_store_ps(&data[i], out);
    } else {
      _mm256_storeu_ps(&data[i], out);
    }
  }

  // Tail fallback for remaining elements
  for (size_t i = vec_size; i < size; i++) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    float u1 = (rng_state & 0xFFFFFF) / float(1 << 24);

    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    float u2 = (rng_state & 0xFFFFFF) / float(1 << 24);

    // Box-Muller: ensure u1 > 0 to avoid log(0)
    u1 = std::max(u1, 1e-8f);
    float r = std::sqrt(-2.0f * std::log(u1));
    float theta = 2.0f * 3.14159265359f * u2;
    data[i] = mean + stddev * r * std::cos(theta);
  }
#else
  // Fallback to standard library if AVX2 not available
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dist(gen);
  }
#endif
}

} // namespace fp
} // namespace cpu
} // namespace ops
} // namespace tnn