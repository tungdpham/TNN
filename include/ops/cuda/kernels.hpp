#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace tunx {
namespace ops {
namespace cuda {

template <typename T>
void add(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void sub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void mul(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void div(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void fmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void fmsub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void fnmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void add_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void sub_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void mul_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void div_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void set_scalar(T *c, T scalar, size_t size, cudaStream_t stream);

template <typename T>
void axpy(T alpha, const T *x, T *y, size_t size, cudaStream_t stream);

template <typename T>
void sqrt(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T>
void rsqrt(const float *a, float *c, size_t size, cudaStream_t stream);

template <typename T>
void rcp(const float *a, float *c, size_t size, cudaStream_t stream);

template <typename T>
void abs(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T>
void min(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void max(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void scalar_max(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void clamp(const T *a, T min_val, T max_val, T *c, size_t size, cudaStream_t stream);

template <typename T>
void equal(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void greater(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void copy(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T>
void h2d_copy(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T>
void d2h_copy(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T>
void zero(T *c, size_t size, cudaStream_t stream);

template <typename T>
T sum(const T *a, size_t size, cudaStream_t stream);

template <typename T>
T dot_product(const T *a, const T *b, size_t size, cudaStream_t stream);

template <typename T>
T norm_squared(const T *a, size_t size, cudaStream_t stream);

template <typename T>
T sum_squared_diff(const T *a, T mean, size_t size, cudaStream_t stream);

template <typename T>
void sub_mul_scalar(const T *a, T sub_scalar, T mul_scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void mul_add_scalar(const T *a, T mul_scalar, T add_scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void fill_random_uniform(T *data, size_t size, T min_val, T max_val, unsigned long long seed,
                         cudaStream_t stream);

template <typename T>
void fill_random_normal(T *data, size_t size, T mean, T stddev, unsigned long long seed,
                        cudaStream_t stream);

template <typename A_T, typename B_T>
void cast(const A_T *a, B_T *b, size_t size, cudaStream_t stream);

template <typename T>
void bswap(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T>
void check_equals(const T *a, const T *b, size_t size, bool &result, double eps,
                  cudaStream_t stream);

}  // namespace cuda
}  // namespace ops
}  // namespace tunx