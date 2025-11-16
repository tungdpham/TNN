#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace tnn {
namespace cuda {

template <typename T> __global__ void fmadd_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T> __global__ void fmsub_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T> __global__ void fnmadd_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T> __global__ void sqrt_kernel(const T *a, T *c, size_t size);
template <typename T> __global__ void abs_kernel(const T *a, T *c, size_t size);
template <typename T> __global__ void min_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T> __global__ void max_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T> __global__ void scalar_max_kernel(const T *a, T scalar, T *c, size_t size);
template <typename T>
__global__ void clamp_kernel(const T *a, T min_val, T max_val, T *c, size_t size);
template <typename T> __global__ void equal_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T> __global__ void greater_kernel(const T *a, const T *b, T *c, size_t size);
template <typename T>
__global__ void fill_random_uniform_kernel(T *data, size_t size, T min_val, T max_val,
                                           unsigned long long seed);
template <typename T>
__global__ void fill_random_normal_kernel(T *data, size_t size, T mean, T stddev,
                                          unsigned long long seed);

} // namespace cuda
} // namespace tnn

#endif // USE_CUDA

#include <cmath>
#include <cstddef>

namespace tnn {
namespace cuda {

// --- Host Function Declarations ---

template <typename T> void cuda_add(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_sub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_mul(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_div(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_fmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_fmsub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_fnmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_add_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_mul_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_div_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_set_scalar(T *c, T scalar, size_t size, cudaStream_t stream);

template <typename T> void cuda_sqrt(const T *a, T *c, size_t size, cudaStream_t stream);

void cuda_rsqrt(const float *a, float *c, size_t size, cudaStream_t stream);
void cuda_rcp(const float *a, float *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_abs(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_min(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_max(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_scalar_max(const T *a, T scalar, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_clamp(const T *a, T min_val, T max_val, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_equal(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T>
void cuda_greater(const T *a, const T *b, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_copy(const T *a, T *c, size_t size, cudaStream_t stream);

template <typename T> void cuda_zero(T *c, size_t size, cudaStream_t stream);

template <typename T> T cuda_sum(const T *a, size_t size, cudaStream_t stream);

template <typename T> T cuda_dot_product(const T *a, const T *b, size_t size, cudaStream_t stream);

template <typename T> T cuda_norm_squared(const T *a, size_t size, cudaStream_t stream);

template <typename T> T cuda_sum_squared_diff(const T *a, T mean, size_t size, cudaStream_t stream);

template <typename T>
void cuda_sub_mul_scalar(const T *a, T sub_scalar, T mul_scalar, T *c, size_t size,
                         cudaStream_t stream);

template <typename T>
void cuda_mul_add_scalar(const T *a, T mul_scalar, T add_scalar, T *c, size_t size,
                         cudaStream_t stream);

template <typename T>
void cuda_fill_random_uniform(T *data, size_t size, T min_val, T max_val, unsigned long long seed,
                              cudaStream_t stream);

template <typename T>
void cuda_fill_random_normal(T *data, size_t size, T mean, T stddev, unsigned long long seed,
                             cudaStream_t stream);

template <typename T>
void cuda_transpose_2d(const T *input, T *output, size_t rows, size_t cols, cudaStream_t stream);

template <typename T>
void cuda_nchw_to_cnhw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream);

template <typename T>
void cuda_cnhw_to_nchw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream);

} // namespace cuda
} // namespace tnn