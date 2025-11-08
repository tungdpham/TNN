#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include <cmath>
#include <cstddef>

namespace cuda {

// Basic arithmetic operations
void cuda_add(const float *a, const float *b, float *c, size_t size);
void cuda_sub(const float *a, const float *b, float *c, size_t size);
void cuda_mul(const float *a, const float *b, float *c, size_t size);
void cuda_div(const float *a, const float *b, float *c, size_t size);

void cuda_add(const double *a, const double *b, double *c, size_t size);
void cuda_sub(const double *a, const double *b, double *c, size_t size);
void cuda_mul(const double *a, const double *b, double *c, size_t size);
void cuda_div(const double *a, const double *b, double *c, size_t size);

// Fused multiply-add operations
void cuda_fmadd(const float *a, const float *b, float *c, size_t size);
void cuda_fmsub(const float *a, const float *b, float *c, size_t size);
void cuda_fnmadd(const float *a, const float *b, float *c, size_t size);

void cuda_fmadd(const double *a, const double *b, double *c, size_t size);
void cuda_fmsub(const double *a, const double *b, double *c, size_t size);
void cuda_fnmadd(const double *a, const double *b, double *c, size_t size);

// Scalar operations
void cuda_add_scalar(const float *a, float scalar, float *c, size_t size);
void cuda_mul_scalar(const float *a, float scalar, float *c, size_t size);
void cuda_div_scalar(const float *a, float scalar, float *c, size_t size);
void cuda_set_scalar(float *c, float scalar, size_t size);

void cuda_add_scalar(const double *a, double scalar, double *c, size_t size);
void cuda_mul_scalar(const double *a, double scalar, double *c, size_t size);
void cuda_div_scalar(const double *a, double scalar, double *c, size_t size);
void cuda_set_scalar(double *c, double scalar, size_t size);

// Mathematical functions
void cuda_sqrt(const float *a, float *c, size_t size);
void cuda_rsqrt(const float *a, float *c, size_t size);
void cuda_rcp(const float *a, float *c, size_t size);
void cuda_abs(const float *a, float *c, size_t size);

void cuda_sqrt(const double *a, double *c, size_t size);
void cuda_abs(const double *a, double *c, size_t size);

// Min/Max operations
void cuda_min(const float *a, const float *b, float *c, size_t size);
void cuda_max(const float *a, const float *b, float *c, size_t size);
void cuda_scalar_max(const float *a, float scalar, float *c, size_t size);
void cuda_clamp(const float *a, float min_val, float max_val, float *c, size_t size);

void cuda_min(const double *a, const double *b, double *c, size_t size);
void cuda_max(const double *a, const double *b, double *c, size_t size);
void cuda_scalar_max(const double *a, double scalar, double *c, size_t size);
void cuda_clamp(const double *a, double min_val, double max_val, double *c, size_t size);

// Comparison operations
void cuda_equal(const float *a, const float *b, float *c, size_t size);
void cuda_greater(const float *a, const float *b, float *c, size_t size);

void cuda_equal(const double *a, const double *b, double *c, size_t size);
void cuda_greater(const double *a, const double *b, double *c, size_t size);

// Memory operations
void cuda_copy(const float *a, float *c, size_t size);
void cuda_zero(float *c, size_t size);

void cuda_copy(const double *a, double *c, size_t size);
void cuda_zero(double *c, size_t size);

// Reduction operations
float cuda_sum(const float *a, size_t size);
float cuda_dot_product(const float *a, const float *b, size_t size);
float cuda_norm_squared(const float *a, size_t size);
float cuda_sum_squared_diff(const float *a, float mean, size_t size);

double cuda_sum(const double *a, size_t size);
double cuda_dot_product(const double *a, const double *b, size_t size);
double cuda_norm_squared(const double *a, size_t size);
double cuda_sum_squared_diff(const double *a, double mean, size_t size);

// Specialized BatchNorm operations
void cuda_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c, size_t size);
void cuda_mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c, size_t size);

void cuda_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c,
                         size_t size);
void cuda_mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c,
                         size_t size);

} // namespace cuda
