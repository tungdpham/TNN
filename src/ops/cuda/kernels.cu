#include "ops/cuda/kernels.hpp"

#ifdef USE_CUDA

#include "cuda/error_handler.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <iostream>
namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(size_t size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

template <typename T> __global__ void add_kernel(const T *a, const T *b, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

template <typename T> __global__ void sub_kernel(const T *a, const T *b, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] - b[idx];
  }
}

template <typename T> __global__ void mul_kernel(const T *a, const T *b, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * b[idx];
  }
}

template <typename T> __global__ void div_kernel(const T *a, const T *b, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / b[idx];
  }
}

template <>
__global__ void fmadd_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(a[idx], b[idx], c[idx]);
  }
}

template <>
__global__ void fmsub_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(a[idx], b[idx], -c[idx]);
  }
}

template <>
__global__ void fnmadd_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(-a[idx], b[idx], c[idx]);
  }
}

template <>
__global__ void fmadd_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(a[idx], b[idx], c[idx]);
  }
}

template <>
__global__ void fmsub_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(a[idx], b[idx], -c[idx]);
  }
}

template <>
__global__ void fnmadd_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(-a[idx], b[idx], c[idx]);
  }
}

template <typename T> __global__ void add_scalar_kernel(const T *a, T scalar, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + scalar;
  }
}

template <typename T> __global__ void sub_scalar_kernel(const T *a, T scalar, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] - scalar;
  }
}

template <typename T> __global__ void mul_scalar_kernel(const T *a, T scalar, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * scalar;
  }
}

template <typename T> __global__ void div_scalar_kernel(const T *a, T scalar, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / scalar;
  }
}

template <typename T> __global__ void set_scalar_kernel(T *c, T scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = scalar;
  }
}

template <> __global__ void sqrt_kernel<float>(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = sqrtf(a[idx]);
  }
}

__global__ void rsqrt_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = rsqrtf(a[idx]);
  }
}

__global__ void rcp_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = 1.0f / a[idx];
  }
}

template <> __global__ void abs_kernel<float>(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fabsf(a[idx]);
  }
}

template <> __global__ void sqrt_kernel<double>(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = sqrt(a[idx]);
  }
}

template <> __global__ void abs_kernel<double>(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fabs(a[idx]);
  }
}

template <>
__global__ void min_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fminf(a[idx], b[idx]);
  }
}

template <>
__global__ void max_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(a[idx], b[idx]);
  }
}

template <>
__global__ void scalar_max_kernel<float>(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(a[idx], scalar);
  }
}

template <>
__global__ void clamp_kernel<float>(const float *a, float min_val, float max_val, float *c,
                                    size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(min_val, fminf(max_val, a[idx]));
  }
}

template <>
__global__ void min_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmin(a[idx], b[idx]);
  }
}

template <>
__global__ void max_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(a[idx], b[idx]);
  }
}

template <>
__global__ void scalar_max_kernel<double>(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(a[idx], scalar);
  }
}

template <>
__global__ void clamp_kernel<double>(const double *a, double min_val, double max_val, double *c,
                                     size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(min_val, fmin(max_val, a[idx]));
  }
}

template <>
__global__ void equal_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
  }
}

template <>
__global__ void greater_kernel<float>(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
  }
}

template <>
__global__ void equal_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
  }
}

template <>
__global__ void greater_kernel<double>(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] > b[idx]) ? 1.0 : 0.0;
  }
}

template <typename T> __global__ void copy_kernel(const T *a, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx];
  }
}

template <typename T> __global__ void zero_kernel(T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (T)0.0;
  }
}

template <typename T>
__global__ void sub_mul_scalar_kernel(const T *a, T sub_scalar, T mul_scalar, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] - sub_scalar) * mul_scalar;
  }
}

template <typename T>
__global__ void mul_add_scalar_kernel(const T *a, T mul_scalar, T add_scalar, T *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * mul_scalar + add_scalar;
  }
}

template <typename T> __global__ void sum_kernel(const T *a, T *result, size_t size) {
  extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
  T *sdata = reinterpret_cast<T *>(shared_mem);

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? a[idx] : (T)0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

template <typename T>
__global__ void dot_product_kernel(const T *a, const T *b, T *result, size_t size) {
  extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
  T *sdata = reinterpret_cast<T *>(shared_mem);

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? a[idx] * b[idx] : (T)0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

template <typename T>
__global__ void sum_squared_diff_kernel(const T *a, T mean, T *result, size_t size) {
  extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
  T *sdata = reinterpret_cast<T *>(shared_mem);

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    T diff = a[idx] - mean;
    sdata[tid] = diff * diff;
  } else {
    sdata[tid] = (T)0.0;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

template <>
__global__ void fill_random_uniform_kernel<float>(float *data, size_t size, float min_val,
                                                  float max_val, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    float random_val = curand_uniform(&state);
    data[idx] = min_val + random_val * (max_val - min_val);
  }
}

template <>
__global__ void fill_random_normal_kernel<float>(float *data, size_t size, float mean, float stddev,
                                                 unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    data[idx] = mean + stddev * curand_normal(&state);
  }
}

template <>
__global__ void fill_random_uniform_kernel<double>(double *data, size_t size, double min_val,
                                                   double max_val, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    double random_val = curand_uniform_double(&state);
    data[idx] = min_val + random_val * (max_val - min_val);
  }
}

template <>
__global__ void fill_random_normal_kernel<double>(double *data, size_t size, double mean,
                                                  double stddev, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    data[idx] = mean + stddev * curand_normal_double(&state);
  }
}

template <typename T>
__global__ void transpose_2d_kernel(const T *input, T *output, size_t rows, size_t cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    int input_idx = row * cols + col;
    int output_idx = col * rows + row;
    output[output_idx] = input[input_idx];
  }
}

template <typename T>
__global__ void nchw_to_cnhw_kernel(const T *input, T *output, size_t n, size_t c, size_t h,
                                    size_t w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = n * c * h * w;

  if (idx < total_size) {
    int n_idx = idx / (c * h * w);
    int remaining = idx % (c * h * w);
    int c_idx = remaining / (h * w);
    remaining = remaining % (h * w);
    int h_idx = remaining / w;
    int w_idx = remaining % w;

    int output_idx = c_idx * (n * h * w) + n_idx * (h * w) + h_idx * w + w_idx;
    output[output_idx] = input[idx];
  }
}

template <typename T>
__global__ void cnhw_to_nchw_kernel(const T *input, T *output, size_t n, size_t c, size_t h,
                                    size_t w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = n * c * h * w;

  if (idx < total_size) {
    int c_idx = idx / (n * h * w);
    int remaining = idx % (n * h * w);
    int n_idx = remaining / (h * w);
    remaining = remaining % (h * w);
    int h_idx = remaining / w;
    int w_idx = remaining % w;

    int output_idx = n_idx * (c * h * w) + c_idx * (h * w) + h_idx * w + w_idx;
    output[output_idx] = input[idx];
  }
}

template <typename T>
void cuda_add(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_sub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  sub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_mul(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  mul_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_div(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  div_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_fmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  fmadd_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_fmsub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  fmsub_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_fnmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  fnmadd_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_add_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  add_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_sub_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  sub_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_mul_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  mul_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_div_scalar(const T *a, T scalar, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  div_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T> void cuda_set_scalar(T *c, T scalar, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  set_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(c, scalar, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T> void cuda_sqrt(const T *a, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  sqrt_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_rsqrt(const float *a, float *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  rsqrt_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_rcp(const float *a, float *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  rcp_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T> void cuda_abs(const T *a, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  abs_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_min(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  min_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_max(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  max_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_scalar_max(const T *a, T scalar, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  scalar_max_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_clamp(const T *a, T min_val, T max_val, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  clamp_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, min_val, max_val, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_equal(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  equal_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_greater(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  greater_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T> void cuda_copy(const T *a, T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  cudaMemcpyAsync(c, a, size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T> void cuda_zero(T *c, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return;
  cudaMemsetAsync(c, 0, size * sizeof(T), stream);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T> T cuda_sum(const T *a, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return (T)0.0;
  int num_blocks = get_num_blocks(size);

  T *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(T));

  sum_kernel<T><<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(a, d_temp_result, size);

  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    T *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<T><<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(
          current_input, d_temp_result, remaining);
      current_input = d_temp_result;
      remaining = blocks_needed;
    }
  }

  T result;

  cudaMemcpy(&result, d_temp_result, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

template <typename T> T cuda_dot_product(const T *a, const T *b, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return (T)0.0;
  int num_blocks = get_num_blocks(size);

  T *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(T));

  dot_product_kernel<T>
      <<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(a, b, d_temp_result, size);

  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    T *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<T><<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(
          current_input, d_temp_result, remaining);
      current_input = d_temp_result;
      remaining = blocks_needed;
    }
  }

  T result;
  cudaMemcpy(&result, d_temp_result, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

template <typename T> T cuda_norm_squared(const T *a, size_t size, cudaStream_t stream) {
  return cuda_dot_product(a, a, size, stream);
}

template <typename T>
T cuda_sum_squared_diff(const T *a, T mean, size_t size, cudaStream_t stream) {
  if (size <= 0)
    return (T)0.0;
  int num_blocks = get_num_blocks(size);

  T *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(T));

  sum_squared_diff_kernel<T>
      <<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(a, mean, d_temp_result, size);

  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    T *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<T><<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(
          current_input, d_temp_result, remaining);
      current_input = d_temp_result;
      remaining = blocks_needed;
    }
  }

  T result;
  cudaMemcpy(&result, d_temp_result, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

template <typename T>
void cuda_sub_mul_scalar(const T *a, T sub_scalar, T mul_scalar, T *c, size_t size,
                         cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  sub_mul_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, sub_scalar, mul_scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_mul_add_scalar(const T *a, T mul_scalar, T add_scalar, T *c, size_t size,
                         cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  mul_add_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, mul_scalar, add_scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_fill_random_uniform(T *data, size_t size, T min_val, T max_val, unsigned long long seed,
                              cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  fill_random_uniform_kernel<T>
      <<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, min_val, max_val, seed);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_fill_random_normal(T *data, size_t size, T mean, T stddev, unsigned long long seed,
                             cudaStream_t stream) {
  if (size <= 0)
    return;
  int num_blocks = get_num_blocks(size);
  fill_random_normal_kernel<T>
      <<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, mean, stddev, seed);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_transpose_2d(const T *input, T *output, size_t rows, size_t cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0)
    return;
  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_2d_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_nchw_to_cnhw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream) {
  if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
    return;
  size_t total_size = n * c * h * w;
  int num_blocks = get_num_blocks(total_size);
  nchw_to_cnhw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n, c, h, w);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_cnhw_to_nchw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream) {
  if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
    return;
  size_t total_size = n * c * h * w;
  int num_blocks = get_num_blocks(total_size);
  cnhw_to_nchw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n, c, h, w);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template void cuda_add<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_add<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_sub<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_sub<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_mul<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_mul<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_div<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_div<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_fmadd<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_fmadd<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_fmsub<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_fmsub<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_fnmadd<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_fnmadd<double>(const double *, const double *, double *, size_t, cudaStream_t);

template void cuda_add_scalar<float>(const float *, float, float *, size_t, cudaStream_t);
template void cuda_add_scalar<double>(const double *, double, double *, size_t, cudaStream_t);
template void cuda_sub_scalar<float>(const float *, float, float *, size_t, cudaStream_t);
template void cuda_sub_scalar<double>(const double *, double, double *, size_t, cudaStream_t);
template void cuda_mul_scalar<float>(const float *, float, float *, size_t, cudaStream_t);
template void cuda_mul_scalar<double>(const double *, double, double *, size_t, cudaStream_t);
template void cuda_div_scalar<float>(const float *, float, float *, size_t, cudaStream_t);
template void cuda_div_scalar<double>(const double *, double, double *, size_t, cudaStream_t);
template void cuda_set_scalar<float>(float *, float, size_t, cudaStream_t);
template void cuda_set_scalar<double>(double *, double, size_t, cudaStream_t);

template void cuda_sqrt<float>(const float *, float *, size_t, cudaStream_t);
template void cuda_sqrt<double>(const double *, double *, size_t, cudaStream_t);
template void cuda_abs<float>(const float *, float *, size_t, cudaStream_t);
template void cuda_abs<double>(const double *, double *, size_t, cudaStream_t);

template void cuda_min<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_min<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_max<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_max<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_scalar_max<float>(const float *, float, float *, size_t, cudaStream_t);
template void cuda_scalar_max<double>(const double *, double, double *, size_t, cudaStream_t);
template void cuda_clamp<float>(const float *, float, float, float *, size_t, cudaStream_t);
template void cuda_clamp<double>(const double *, double, double, double *, size_t, cudaStream_t);

template void cuda_equal<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_equal<double>(const double *, const double *, double *, size_t, cudaStream_t);
template void cuda_greater<float>(const float *, const float *, float *, size_t, cudaStream_t);
template void cuda_greater<double>(const double *, const double *, double *, size_t, cudaStream_t);

template void cuda_copy<float>(const float *, float *, size_t, cudaStream_t);
template void cuda_copy<double>(const double *, double *, size_t, cudaStream_t);
template void cuda_zero<float>(float *, size_t, cudaStream_t);
template void cuda_zero<double>(double *, size_t, cudaStream_t);

template float cuda_sum<float>(const float *, size_t, cudaStream_t);
template double cuda_sum<double>(const double *, size_t, cudaStream_t);
template float cuda_dot_product<float>(const float *, const float *, size_t, cudaStream_t);
template double cuda_dot_product<double>(const double *, const double *, size_t, cudaStream_t);
template float cuda_norm_squared<float>(const float *, size_t, cudaStream_t);
template double cuda_norm_squared<double>(const double *, size_t, cudaStream_t);
template float cuda_sum_squared_diff<float>(const float *, float, size_t, cudaStream_t);
template double cuda_sum_squared_diff<double>(const double *, double, size_t, cudaStream_t);

template void cuda_sub_mul_scalar<float>(const float *, float, float, float *, size_t,
                                         cudaStream_t);
template void cuda_sub_mul_scalar<double>(const double *, double, double, double *, size_t,
                                          cudaStream_t);
template void cuda_mul_add_scalar<float>(const float *, float, float, float *, size_t,
                                         cudaStream_t);
template void cuda_mul_add_scalar<double>(const double *, double, double, double *, size_t,
                                          cudaStream_t);

template void cuda_fill_random_uniform<float>(float *data, size_t size, float min_val,
                                              float max_val, unsigned long long seed,
                                              cudaStream_t stream);
template void cuda_fill_random_uniform<double>(double *data, size_t size, double min_val,
                                               double max_val, unsigned long long seed,
                                               cudaStream_t stream);
template void cuda_fill_random_normal<float>(float *data, size_t size, float mean, float stddev,
                                             unsigned long long seed, cudaStream_t stream);
template void cuda_fill_random_normal<double>(double *data, size_t size, double mean, double stddev,
                                              unsigned long long seed, cudaStream_t stream);

template void cuda_transpose_2d<float>(const float *input, float *output, size_t rows, size_t cols,
                                       cudaStream_t stream);
template void cuda_transpose_2d<double>(const double *input, double *output, size_t rows,
                                        size_t cols, cudaStream_t stream);
template void cuda_nchw_to_cnhw<float>(const float *input, float *output, size_t n, size_t c,
                                       size_t h, size_t w, cudaStream_t stream);
template void cuda_nchw_to_cnhw<double>(const double *input, double *output, size_t n, size_t c,
                                        size_t h, size_t w, cudaStream_t stream);
template void cuda_cnhw_to_nchw<float>(const float *input, float *output, size_t n, size_t c,
                                       size_t h, size_t w, cudaStream_t stream);
template void cuda_cnhw_to_nchw<double>(const double *input, double *output, size_t n, size_t c,
                                        size_t h, size_t w, cudaStream_t stream);

} // namespace cuda

#endif
}