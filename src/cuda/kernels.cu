#include "cuda/kernels.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {

// Kernel execution configuration helper
constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(size_t size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

// Basic arithmetic kernels for float
__global__ void add_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void sub_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] - b[idx];
  }
}

__global__ void mul_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * b[idx];
  }
}

__global__ void div_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / b[idx];
  }
}

// Basic arithmetic kernels for double
__global__ void add_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void sub_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] - b[idx];
  }
}

__global__ void mul_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * b[idx];
  }
}

__global__ void div_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / b[idx];
  }
}

// Fused multiply-add kernels for float
__global__ void fmadd_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(a[idx], b[idx], c[idx]);
  }
}

__global__ void fmsub_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(a[idx], b[idx], -c[idx]);
  }
}

__global__ void fnmadd_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(-a[idx], b[idx], c[idx]);
  }
}

// Fused multiply-add kernels for double
__global__ void fmadd_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(a[idx], b[idx], c[idx]);
  }
}

__global__ void fmsub_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(a[idx], b[idx], -c[idx]);
  }
}

__global__ void fnmadd_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(-a[idx], b[idx], c[idx]);
  }
}

// Scalar operation kernels for float
__global__ void add_scalar_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + scalar;
  }
}

__global__ void mul_scalar_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * scalar;
  }
}

__global__ void div_scalar_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / scalar;
  }
}

__global__ void set_scalar_kernel(float *c, float scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = scalar;
  }
}

// Scalar operation kernels for double
__global__ void add_scalar_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + scalar;
  }
}

__global__ void mul_scalar_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * scalar;
  }
}

__global__ void div_scalar_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / scalar;
  }
}

__global__ void set_scalar_kernel(double *c, double scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = scalar;
  }
}

// Mathematical function kernels for float
__global__ void sqrt_kernel(const float *a, float *c, size_t size) {
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

__global__ void abs_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fabsf(a[idx]);
  }
}

// Mathematical function kernels for double
__global__ void sqrt_kernel(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = sqrt(a[idx]);
  }
}

__global__ void abs_kernel(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fabs(a[idx]);
  }
}

// Min/Max operation kernels for float
__global__ void min_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fminf(a[idx], b[idx]);
  }
}

__global__ void max_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(a[idx], b[idx]);
  }
}

__global__ void scalar_max_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(a[idx], scalar);
  }
}

__global__ void clamp_kernel(const float *a, float min_val, float max_val, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(min_val, fminf(max_val, a[idx]));
  }
}

// Min/Max operation kernels for double
__global__ void min_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmin(a[idx], b[idx]);
  }
}

__global__ void max_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(a[idx], b[idx]);
  }
}

__global__ void scalar_max_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(a[idx], scalar);
  }
}

__global__ void clamp_kernel(const double *a, double min_val, double max_val, double *c,
                             size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(min_val, fmin(max_val, a[idx]));
  }
}

// Comparison operation kernels for float
__global__ void equal_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
  }
}

__global__ void greater_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
  }
}

// Comparison operation kernels for double
__global__ void equal_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
  }
}

__global__ void greater_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] > b[idx]) ? 1.0 : 0.0;
  }
}

// Memory operation kernels for float
__global__ void copy_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx];
  }
}

__global__ void zero_kernel(float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = 0.0f;
  }
}

// Memory operation kernels for double
__global__ void copy_kernel(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx];
  }
}

__global__ void zero_kernel(double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = 0.0;
  }
}

// Specialized BatchNorm operation kernels for float
__global__ void sub_mul_scalar_kernel(const float *a, float sub_scalar, float mul_scalar, float *c,
                                      size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] - sub_scalar) * mul_scalar;
  }
}

__global__ void mul_add_scalar_kernel(const float *a, float mul_scalar, float add_scalar, float *c,
                                      size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * mul_scalar + add_scalar;
  }
}

// Specialized BatchNorm operation kernels for double
__global__ void sub_mul_scalar_kernel(const double *a, double sub_scalar, double mul_scalar,
                                      double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] - sub_scalar) * mul_scalar;
  }
}

__global__ void mul_add_scalar_kernel(const double *a, double mul_scalar, double add_scalar,
                                      double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * mul_scalar + add_scalar;
  }
}

// Reduction operation kernels for float
__global__ void sum_kernel(const float *a, float *result, size_t size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (idx < size) ? a[idx] : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

__global__ void dot_product_kernel(const float *a, const float *b, float *result, size_t size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory and compute product
  sdata[tid] = (idx < size) ? a[idx] * b[idx] : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

__global__ void sum_squared_diff_kernel(const float *a, float mean, float *result, size_t size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory and compute squared difference
  if (idx < size) {
    float diff = a[idx] - mean;
    sdata[tid] = diff * diff;
  } else {
    sdata[tid] = 0.0f;
  }
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

// Reduction operation kernels for double
__global__ void sum_kernel(const double *a, double *result, size_t size) {
  extern __shared__ double sdata_double[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata_double[tid] = (idx < size) ? a[idx] : 0.0;
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_double[tid] += sdata_double[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = sdata_double[0];
  }
}

__global__ void dot_product_kernel(const double *a, const double *b, double *result, size_t size) {
  extern __shared__ double sdata_double[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory and compute product
  sdata_double[tid] = (idx < size) ? a[idx] * b[idx] : 0.0;
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_double[tid] += sdata_double[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = sdata_double[0];
  }
}

__global__ void sum_squared_diff_kernel(const double *a, double mean, double *result, size_t size) {
  extern __shared__ double sdata_double[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory and compute squared difference
  if (idx < size) {
    double diff = a[idx] - mean;
    sdata_double[tid] = diff * diff;
  } else {
    sdata_double[tid] = 0.0;
  }
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_double[tid] += sdata_double[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = sdata_double[0];
  }
}

void cuda_add(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  add_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_sub(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  sub_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_mul(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  mul_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_div(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  div_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

// Host wrapper functions - Basic arithmetic operations for double
void cuda_add(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  add_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_sub(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  sub_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_mul(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  mul_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_div(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  div_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

// Host wrapper functions - Fused multiply-add operations for float
void cuda_fmadd(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  fmadd_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_fmsub(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  fmsub_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_fnmadd(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  fnmadd_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

// Host wrapper functions - Fused multiply-add operations for double
void cuda_fmadd(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  fmadd_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_fmsub(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  fmsub_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_fnmadd(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  fnmadd_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

// Host wrapper functions - Scalar operations for float
void cuda_add_scalar(const float *a, float scalar, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  add_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_mul_scalar(const float *a, float scalar, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  mul_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_div_scalar(const float *a, float scalar, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  div_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_set_scalar(float *c, float scalar, size_t size) {
  int num_blocks = get_num_blocks(size);
  set_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(c, scalar, size);
}

// Host wrapper functions - Scalar operations for double
void cuda_add_scalar(const double *a, double scalar, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  add_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_mul_scalar(const double *a, double scalar, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  mul_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_div_scalar(const double *a, double scalar, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  div_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_set_scalar(double *c, double scalar, size_t size) {
  int num_blocks = get_num_blocks(size);
  set_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(c, scalar, size);
}

// Host wrapper functions - Mathematical functions for float
void cuda_sqrt(const float *a, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  sqrt_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

void cuda_rsqrt(const float *a, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  rsqrt_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

void cuda_rcp(const float *a, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  rcp_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

void cuda_abs(const float *a, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  abs_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

// Host wrapper functions - Mathematical functions for double
void cuda_sqrt(const double *a, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  sqrt_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

void cuda_abs(const double *a, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  abs_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

// Host wrapper functions - Min/Max operations for float
void cuda_min(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  min_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_max(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  max_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_scalar_max(const float *a, float scalar, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  scalar_max_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_clamp(const float *a, float min_val, float max_val, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  clamp_kernel<<<num_blocks, BLOCK_SIZE>>>(a, min_val, max_val, c, size);
}

// Host wrapper functions - Min/Max operations for double
void cuda_min(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  min_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_max(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  max_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_scalar_max(const double *a, double scalar, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  scalar_max_kernel<<<num_blocks, BLOCK_SIZE>>>(a, scalar, c, size);
}

void cuda_clamp(const double *a, double min_val, double max_val, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  clamp_kernel<<<num_blocks, BLOCK_SIZE>>>(a, min_val, max_val, c, size);
}

// Host wrapper functions - Comparison operations for float
void cuda_equal(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  equal_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_greater(const float *a, const float *b, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  greater_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

// Host wrapper functions - Comparison operations for double
void cuda_equal(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  equal_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

void cuda_greater(const double *a, const double *b, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  greater_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, c, size);
}

// Host wrapper functions - Memory operations for float
void cuda_copy(const float *a, float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  copy_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

void cuda_zero(float *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  zero_kernel<<<num_blocks, BLOCK_SIZE>>>(c, size);
}

// Host wrapper functions - Memory operations for double
void cuda_copy(const double *a, double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  copy_kernel<<<num_blocks, BLOCK_SIZE>>>(a, c, size);
}

void cuda_zero(double *c, size_t size) {
  int num_blocks = get_num_blocks(size);
  zero_kernel<<<num_blocks, BLOCK_SIZE>>>(c, size);
}

// Host wrapper functions - Reduction operations for float
float cuda_sum(const float *a, size_t size) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  float *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(float));

  // Launch kernel
  sum_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(a, d_temp_result, size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    // Recursively reduce until we have a single value
    size_t remaining = num_blocks;
    float *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host
  float result;
  cudaMemcpy(&result, d_temp_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);

  return result;
}

float cuda_dot_product(const float *a, const float *b, size_t size) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  float *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(float));

  // Launch kernel
  dot_product_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(a, b, d_temp_result,
                                                                             size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    // Recursively reduce until we have a single value
    size_t remaining = num_blocks;
    float *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host
  float result;
  cudaMemcpy(&result, d_temp_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);

  return result;
}

float cuda_norm_squared(const float *a, size_t size) { return cuda_dot_product(a, a, size); }

float cuda_sum_squared_diff(const float *a, float mean, size_t size) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  float *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(float));

  // Launch kernel
  sum_squared_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
      a, mean, d_temp_result, size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    // Recursively reduce until we have a single value
    size_t remaining = num_blocks;
    float *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host
  float result;
  cudaMemcpy(&result, d_temp_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);

  return result;
}

// Host wrapper functions - Reduction operations for double
double cuda_sum(const double *a, size_t size) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  double *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(double));

  // Launch kernel
  sum_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(a, d_temp_result, size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    // Recursively reduce until we have a single value
    size_t remaining = num_blocks;
    double *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host
  double result;
  cudaMemcpy(&result, d_temp_result, sizeof(double), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);

  return result;
}

double cuda_dot_product(const double *a, const double *b, size_t size) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  double *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(double));

  // Launch kernel
  dot_product_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(a, b, d_temp_result,
                                                                              size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    // Recursively reduce until we have a single value
    size_t remaining = num_blocks;
    double *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host
  double result;
  cudaMemcpy(&result, d_temp_result, sizeof(double), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);

  return result;
}

double cuda_norm_squared(const double *a, size_t size) { return cuda_dot_product(a, a, size); }

double cuda_sum_squared_diff(const double *a, double mean, size_t size) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  double *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(double));

  // Launch kernel
  sum_squared_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
      a, mean, d_temp_result, size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    // Recursively reduce until we have a single value
    size_t remaining = num_blocks;
    double *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host
  double result;
  cudaMemcpy(&result, d_temp_result, sizeof(double), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);

  return result;
}

void cuda_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c,
                         size_t size) {
  int num_blocks = get_num_blocks(size);
  sub_mul_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, sub_scalar, mul_scalar, c, size);
}

void cuda_mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c,
                         size_t size) {
  int num_blocks = get_num_blocks(size);
  mul_add_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, mul_scalar, add_scalar, c, size);
}

void cuda_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c,
                         size_t size) {
  int num_blocks = get_num_blocks(size);
  sub_mul_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, sub_scalar, mul_scalar, c, size);
}

void cuda_mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c,
                         size_t size) {
  int num_blocks = get_num_blocks(size);
  mul_add_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(a, mul_scalar, add_scalar, c, size);
}

} // namespace cuda

#endif // USE_CUDA
