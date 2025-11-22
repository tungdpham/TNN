#pragma once

#include <cmath>
#include <cstddef> // For size_t

namespace tnn {
namespace ops {
namespace cpu {
namespace fp {
// Single-precision AVX2 Implementations
#ifdef __AVX2__

// Basic Arithmetic (Float)
inline void avx2_unaligned_add(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_add(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_sub(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_sub(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_mul(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_mul(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_div(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_div(const float *a, const float *b, float *c, size_t size);

// Fused Multiply-Add Operations (Float)
inline void avx2_unaligned_fmadd(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_fmadd(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_fmsub(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_fmsub(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_fnmadd(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_fnmadd(const float *a, const float *b, float *c, size_t size);

// Scalar Operations (Float)
inline void avx2_unaligned_add_scalar(const float *a, float scalar, float *c, size_t size);
inline void avx2_aligned_add_scalar(const float *a, float scalar, float *c, size_t size);
inline void avx2_unaligned_sub_scalar(const float *a, float scalar, float *c, size_t size);
inline void avx2_aligned_sub_scalar(const float *a, float scalar, float *c, size_t size);
inline void avx2_unaligned_mul_scalar(const float *a, float scalar, float *c, size_t size);
inline void avx2_aligned_mul_scalar(const float *a, float scalar, float *c, size_t size);
inline void avx2_unaligned_div_scalar(const float *a, const float scalar, float *c, size_t size);
inline void avx2_aligned_div_scalar(const float *a, const float scalar, float *c, size_t size);
inline void avx2_unaligned_set_scalar(float *c, float scalar, size_t size);
inline void avx2_aligned_set_scalar(float *c, float scalar, size_t size);

// Element-wise Functions (Float)
inline void avx2_unaligned_sqrt(const float *a, float *c, size_t size);
inline void avx2_aligned_sqrt(const float *a, float *c, size_t size);
inline void avx2_unaligned_rsqrt(const float *a, float *c, size_t size);
inline void avx2_aligned_rsqrt(const float *a, float *c, size_t size);
inline void avx2_unaligned_rcp(const float *a, float *c, size_t size);
inline void avx2_aligned_rcp(const float *a, float *c, size_t size);
inline void avx2_unaligned_abs(const float *a, float *c, size_t size);
inline void avx2_aligned_abs(const float *a, float *c, size_t size);

// Comparison and Clamping (Float)
inline void avx2_unaligned_min(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_min(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_max(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_max(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_scalar_max(const float *a, float scalar, float *c, size_t size);
inline void avx2_aligned_scalar_max(const float *a, float scalar, float *c, size_t size);
inline void avx2_unaligned_clamp(const float *a, float min_val, float max_val, float *c,
                                 size_t size);
inline void avx2_aligned_clamp(const float *a, float min_val, float max_val, float *c, size_t size);
inline void avx2_unaligned_equal(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_equal(const float *a, const float *b, float *c, size_t size);
inline void avx2_unaligned_greater(const float *a, const float *b, float *c, size_t size);
inline void avx2_aligned_greater(const float *a, const float *b, float *c, size_t size);

// Memory and Reduction (Float)
inline void avx2_unaligned_copy(const float *a, float *c, size_t size);
inline void avx2_aligned_copy(const float *a, float *c, size_t size);
inline void avx2_unaligned_zero(float *c, size_t size);
inline void avx2_aligned_zero(float *c, size_t size);
inline float avx2_unaligned_sum(const float *a, size_t size);
inline float avx2_aligned_sum(const float *a, size_t size);
inline float avx2_unaligned_dot_product(const float *a, const float *b, size_t size);
inline float avx2_aligned_dot_product(const float *a, const float *b, size_t size);
inline float avx2_unaligned_sum_squared_diff(const float *a, float mean, size_t size);
inline float avx2_aligned_sum_squared_diff(const float *a, float mean, size_t size);

// Specialized BatchNorm Operations (Float)
inline void avx2_unaligned_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar,
                                          float *c, size_t size);
inline void avx2_aligned_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar,
                                        float *c, size_t size);
inline void avx2_unaligned_mul_add_scalar(const float *a, float mul_scalar, float add_scalar,
                                          float *c, size_t size);
inline void avx2_aligned_mul_add_scalar(const float *a, float mul_scalar, float add_scalar,
                                        float *c, size_t size);

#endif // __AVX2__

// Wrapper Functions (Float)

// Basic Arithmetic (Float)
void add(const float *a, const float *b, float *c, size_t size);
void sub(const float *a, const float *b, float *c, size_t size);
void mul(const float *a, const float *b, float *c, size_t size);
void div(const float *a, const float *b, float *c, size_t size);
void fmadd(const float *a, const float *b, float *c, size_t size);
void fmsub(const float *a, const float *b, float *c, size_t size);
void fnmadd(const float *a, const float *b, float *c, size_t size);

void add_scalar(const float *a, float scalar, float *c, size_t size);
void sub_scalar(const float *a, float scalar, float *c, size_t size);
void mul_scalar(const float *a, float scalar, float *c, size_t size);
void div_scalar(const float *a, float scalar, float *c, size_t size);
void set_scalar(float *c, float scalar, size_t size);

// BLAS-like Operations
void axpy(float alpha, const float *x, float *y, size_t size);

void sqrt(const float *a, float *c, size_t size);
void rsqrt(const float *a, float *c, size_t size);
void rcp(const float *a, float *c, size_t size);
void abs(const float *a, float *c, size_t size);
void min(const float *a, const float *b, float *c, size_t size);
void max(const float *a, const float *b, float *c, size_t size);
void scalar_max(const float *a, float scalar, float *c, size_t size);
void clamp(const float *a, float min_val, float max_val, float *c, size_t size);

void equal(const float *a, const float *b, float *c, size_t size);
void greater(const float *a, const float *b, float *c, size_t size);

void copy(const float *a, float *c, size_t size);
void zero(float *c, size_t size);

// BatchNorm Specialized Operations (Float)
void sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c, size_t size);
void mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c, size_t size);

// Reduction Functions
float sum(const float *a, size_t size);
float dot_product(const float *a, const float *b, size_t size);
float sum_squared_diff(const float *a, float mean, size_t size);

// Utility
void fill_random_uniform(float *data, size_t size, float min_val, float max_val,
                         unsigned long long seed);
void fill_random_normal(float *data, size_t size, float mean, float stddev,
                        unsigned long long seed);
} // namespace fp
} // namespace cpu
} // namespace ops
} // namespace tnn