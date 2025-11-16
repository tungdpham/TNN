#pragma once

#include <cmath>
#include <cstddef> // For size_t

namespace tnn {
namespace ops {
namespace cpu {
namespace dp {
// Double-precision AVX2 Implementations
#ifdef __AVX2__

// Basic Arithmetic (Double)
inline void avx2_unaligned_add(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_add(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_sub(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_sub(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_mul(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_mul(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_div(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_div(const double *a, const double *b, double *c, size_t size);

// Fused Multiply-Add Operations (Double)
inline void avx2_unaligned_fmadd(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_fmadd(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_fmsub(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_fmsub(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_fnmadd(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_fnmadd(const double *a, const double *b, double *c, size_t size);

// Scalar Operations (Double)
inline void avx2_unaligned_add_scalar(const double *a, double scalar, double *c, size_t size);
inline void avx2_aligned_add_scalar(const double *a, double scalar, double *c, size_t size);
inline void avx2_unaligned_mul_scalar(const double *a, double scalar, double *c, size_t size);
inline void avx2_aligned_mul_scalar(const double *a, double scalar, double *c, size_t size);
inline void avx2_unaligned_div_scalar(const double *a, const double scalar, double *c, size_t size);
inline void avx2_aligned_div_scalar(const double *a, const double scalar, double *c, size_t size);
inline void avx2_unaligned_set_scalar(double *c, double scalar, size_t size);
inline void avx2_aligned_set_scalar(double *c, double scalar, size_t size);

// Element-wise Functions (Double)
inline void avx2_unaligned_sqrt(const double *a, double *c, size_t size);
inline void avx2_aligned_sqrt(const double *a, double *c, size_t size);
inline void avx2_unaligned_abs(const double *a, double *c, size_t size);
inline void avx2_aligned_abs(const double *a, double *c, size_t size);

// Comparison and Clamping (Double)
inline void avx2_unaligned_min(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_min(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_max(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_max(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_scalar_max(const double *a, double scalar, double *c, size_t size);
inline void avx2_aligned_scalar_max(const double *a, double scalar, double *c, size_t size);
inline void avx2_unaligned_clamp(const double *a, double min_val, double max_val, double *c,
                                 size_t size);
inline void avx2_aligned_clamp(const double *a, double min_val, double max_val, double *c,
                               size_t size);
inline void avx2_unaligned_equal(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_equal(const double *a, const double *b, double *c, size_t size);
inline void avx2_unaligned_greater(const double *a, const double *b, double *c, size_t size);
inline void avx2_aligned_greater(const double *a, const double *b, double *c, size_t size);

// Memory and Reduction (Double)
inline void avx2_unaligned_copy(const double *a, double *c, size_t size);
inline void avx2_aligned_copy(const double *a, double *c, size_t size);
inline void avx2_unaligned_zero(double *c, size_t size);
inline void avx2_aligned_zero(double *c, size_t size);
inline double avx2_unaligned_sum(const double *a, size_t size);
inline void avx2_aligned_sum(const double *a, size_t size, double *result);
inline double avx2_unaligned_dot_product(const double *a, const double *b, size_t size);
inline double avx2_aligned_dot_product(const double *a, const double *b, size_t size);
inline double avx2_unaligned_sum_squared_diff(const double *a, double mean, size_t size);
inline double avx2_aligned_sum_squared_diff(const double *a, double mean, size_t size);

// Specialized BatchNorm Operations (Double)
inline void avx2_unaligned_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar,
                                          double *c, size_t size);
inline void avx2_aligned_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar,
                                        double *c, size_t size);
inline void avx2_unaligned_mul_add_scalar(const double *a, double mul_scalar, double add_scalar,
                                          double *c, size_t size);
inline void avx2_aligned_mul_add_scalar(const double *a, double mul_scalar, double add_scalar,
                                        double *c, size_t size);

#endif // __AVX2__

// Wrapper Functions (Double)
void add(const double *a, const double *b, double *c, size_t size);
void sub(const double *a, const double *b, double *c, size_t size);
void mul(const double *a, const double *b, double *c, size_t size);
void div(const double *a, const double *b, double *c, size_t size);
void fmadd(const double *a, const double *b, double *c, size_t size);
void fmsub(const double *a, const double *b, double *c, size_t size);
void fnmadd(const double *a, const double *b, double *c, size_t size);

void add_scalar(const double *a, double scalar, double *c, size_t size);
void mul_scalar(const double *a, double scalar, double *c, size_t size);
void div_scalar(const double *a, double scalar, double *c, size_t size);
void set_scalar(double *c, double scalar, size_t size);

void sqrt(const double *a, double *c, size_t size);
void abs(const double *a, double *c, size_t size);
void min(const double *a, const double *b, double *c, size_t size);
void max(const double *a, const double *b, double *c, size_t size);
void scalar_max(const double *a, double scalar, double *c, size_t size);
void clamp(const double *a, double min_val, double max_val, double *c, size_t size);

void equal(const double *a, const double *b, double *c, size_t size);
void greater(const double *a, const double *b, double *c, size_t size);

void copy(const double *a, double *c, size_t size);
void zero(double *c, size_t size);

// BatchNorm Specialized Operations (Double)
void sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c, size_t size);
void mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c, size_t size);

// Reduction Functions
double sum(const double *a, size_t size);
double dot_product(const double *a, const double *b, size_t size);
double sum_squared_diff(const double *a, double mean, size_t size);

// Utility
void fill_random_uniform(double *data, size_t size, double min_val, double max_val,
                         unsigned long long seed);
void fill_random_normal(double *data, size_t size, double mean, double stddev,
                        unsigned long long seed);

} // namespace dp
} // namespace cpu
} // namespace ops
} // namespace tnn