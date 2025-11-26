#include "ops/cuda/kernels.hpp"

#ifdef USE_CUDA

#include "cuda/error_handler.hpp"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <type_traits>

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int TILE_DIM = 32;

inline int get_num_blocks(size_t size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

template <typename T> struct VectorizedTrait;

template <> struct VectorizedTrait<float> {
  using type = float4;
  static constexpr int size = 4;
};

template <> struct VectorizedTrait<double> {
  using type = double2;
  static constexpr int size = 2;
};

namespace functors {

template <typename T> struct Add {
  __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T> struct Sub {
  __device__ T operator()(T a, T b) const { return a - b; }
};
template <typename T> struct Mul {
  __device__ T operator()(T a, T b) const { return a * b; }
};
template <typename T> struct Div {
  __device__ T operator()(T a, T b) const { return a / b; }
};
template <typename T> struct Min {
  __device__ T operator()(T a, T b) const { return (a < b) ? a : b; }
};
template <typename T> struct Max {
  __device__ T operator()(T a, T b) const { return (a > b) ? a : b; }
};

template <typename T> struct Equal {
  __device__ T operator()(T a, T b) const { return (a == b) ? (T)1 : (T)0; }
};
template <typename T> struct Greater {
  __device__ T operator()(T a, T b) const { return (a > b) ? (T)1 : (T)0; }
};

template <typename T> struct FMAdd;
template <> struct FMAdd<float> {
  __device__ float operator()(float a, float b, float c) const { return fmaf(a, b, c); }
};
template <> struct FMAdd<double> {
  __device__ double operator()(double a, double b, double c) const { return fma(a, b, c); }
};

template <typename T> struct FMSub;
template <> struct FMSub<float> {
  __device__ float operator()(float a, float b, float c) const { return fmaf(a, b, -c); }
};
template <> struct FMSub<double> {
  __device__ double operator()(double a, double b, double c) const { return fma(a, b, -c); }
};

template <typename T> struct FNMAdd;
template <> struct FNMAdd<float> {
  __device__ float operator()(float a, float b, float c) const { return fmaf(-a, b, c); }
};
template <> struct FNMAdd<double> {
  __device__ double operator()(double a, double b, double c) const { return fma(-a, b, c); }
};

template <typename T> struct Sqrt {
  __device__ T operator()(T a) const { return sqrt(a); }
};
template <> struct Sqrt<float> {
  __device__ float operator()(float a) const { return sqrtf(a); }
};

template <typename T> struct Rsqrt {
  __device__ T operator()(T a) const { return (T)1 / sqrt(a); }
};
template <> struct Rsqrt<float> {
  __device__ float operator()(float a) const { return rsqrtf(a); }
};

template <typename T> struct Rcp {
  __device__ T operator()(T a) const { return (T)1 / a; }
};

template <typename T> struct Abs {
  __device__ T operator()(T a) const { return abs(a); }
};
template <> struct Abs<float> {
  __device__ float operator()(float a) const { return fabsf(a); }
};
template <> struct Abs<double> {
  __device__ double operator()(double a) const { return fabs(a); }
};

template <typename T> struct AddScalar {
  T s;
  __device__ T operator()(T a) const { return a + s; }
};
template <typename T> struct SubScalar {
  T s;
  __device__ T operator()(T a) const { return a - s; }
};
template <typename T> struct MulScalar {
  T s;
  __device__ T operator()(T a) const { return a * s; }
};
template <typename T> struct DivScalar {
  T s;
  __device__ T operator()(T a) const { return a / s; }
};
template <typename T> struct ScalarMax {
  T s;
  __device__ T operator()(T a) const { return (a > s) ? a : s; }
};
template <typename T> struct Clamp {
  T min_v, max_v;
  __device__ T operator()(T a) const { return (a < min_v) ? min_v : ((a > max_v) ? max_v : a); }
};

template <typename T> struct SubMulScalar {
  T sub, mul;
  __device__ T operator()(T a) const { return (a - sub) * mul; }
};
template <typename T> struct MulAddScalar {
  T mul, add;
  __device__ T operator()(T a) const { return a * mul + add; }
};
template <typename T> struct Axpy {
  T alpha;
  __device__ T operator()(T x, T y) const { return alpha * x + y; }
};
} // namespace functors

template <typename T, typename Func>
__global__ void binary_op_kernel(const T *__restrict__ a, const T *__restrict__ b,
                                 T *__restrict__ c, size_t size, Func op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  using VecT = typename VectorizedTrait<T>::type;
  constexpr int vec_size = VectorizedTrait<T>::size;
  size_t vec_idx = idx * vec_size;

  if (vec_idx + vec_size <= size) {
    VecT va = reinterpret_cast<const VecT *>(a)[idx];
    VecT vb = reinterpret_cast<const VecT *>(b)[idx];
    VecT vc;
    if constexpr (std::is_same<T, float>::value) {
      vc.x = op(va.x, vb.x);
      vc.y = op(va.y, vb.y);
      vc.z = op(va.z, vb.z);
      vc.w = op(va.w, vb.w);
    } else {
      vc.x = op(va.x, vb.x);
      vc.y = op(va.y, vb.y);
    }
    reinterpret_cast<VecT *>(c)[idx] = vc;
  }
}

template <typename T, typename Func>
__global__ void binary_op_scalar_kernel(const T *a, const T *b, T *c, size_t size, Func op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    c[idx] = op(a[idx], b[idx]);
}

template <typename T, typename Func>
__global__ void unary_op_kernel(const T *__restrict__ a, T *__restrict__ c, size_t size, Func op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  using VecT = typename VectorizedTrait<T>::type;
  constexpr int vec_size = VectorizedTrait<T>::size;
  if (idx * vec_size + vec_size <= size) {
    VecT va = reinterpret_cast<const VecT *>(a)[idx];
    VecT vc;
    if constexpr (std::is_same<T, float>::value) {
      vc.x = op(va.x);
      vc.y = op(va.y);
      vc.z = op(va.z);
      vc.w = op(va.w);
    } else {
      vc.x = op(va.x);
      vc.y = op(va.y);
    }
    reinterpret_cast<VecT *>(c)[idx] = vc;
  }
}

template <typename T, typename Func>
__global__ void unary_op_scalar_kernel(const T *a, T *c, size_t size, Func op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    c[idx] = op(a[idx]);
}

template <typename T> __global__ void set_scalar_kernel(T *c, T scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    c[idx] = scalar;
}

template <typename T> __global__ void axpy_kernel(T alpha, const T *x, T *y, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    y[idx] += alpha * x[idx];
}

template <typename T>
__global__ void transpose_2d_kernel(const T *input, T *output, size_t rows, size_t cols) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (y < rows && x < cols) {
    tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (y < cols && x < rows) {
    output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
__global__ void nchw_cnhw_transpose_tiled(const T *__restrict__ input, T *__restrict__ output,
                                          int N, int C, int HW) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int k = 0; k < HW; ++k) {
    if (y < N && x < C) {
      size_t in_idx = (size_t)y * (C * HW) + (size_t)x * HW + k;
      tile[threadIdx.y][threadIdx.x] = input[in_idx];
    }
    __syncthreads();
    int n_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int c_out = blockIdx.x * TILE_DIM + threadIdx.y;
    if (n_out < N && c_out < C) {
      size_t out_idx = (size_t)c_out * (N * HW) + (size_t)n_out * HW + k;
      output[out_idx] = tile[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
  }
}

template <typename T> __inline__ __device__ T warp_reduce_sum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <typename T, int Mode>
__global__ void reduce_kernel(const T *a, const T *b, T scalar, T *result, size_t size) {
  T sum = 0;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < size; i += stride) {
    T val = (T)0;
    if constexpr (Mode == 0) {
      val = a[i];
    } else if constexpr (Mode == 1) {
      val = a[i] * b[i];
    } else if constexpr (Mode == 2) {
      T diff = a[i] - scalar;
      val = diff * diff;
    }
    sum += val;
  }

  sum = warp_reduce_sum(sum);

  static __shared__ double shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;

  if (lane == 0)
    shared[warp] = (double)sum;
  __syncthreads();

  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? (T)shared[lane] : 0;
  if (warp == 0)
    sum = warp_reduce_sum(sum);

  if (threadIdx.x == 0)
    result[blockIdx.x] = sum;
}

__global__ void fill_random_uniform_kernel(float *data, size_t size, float min_val, float max_val,
                                           unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    data[idx] = min_val + curand_uniform(&state) * (max_val - min_val);
  }
}

__global__ void fill_random_uniform_kernel(double *data, size_t size, double min_val,
                                           double max_val, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    data[idx] = min_val + curand_uniform_double(&state) * (max_val - min_val);
  }
}

template <typename T>
__global__ void fill_random_normal_kernel(T *data, size_t size, T mean, T stddev,
                                          unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {

    if constexpr (std::is_same<T, float>::value) {
      float val = curand_normal(&state);
      data[i] = mean + stddev * val;
    } else {
      double val = curand_normal_double(&state);
      data[i] = mean + stddev * val;
    }
  }
}

template <>
__global__ void fill_random_normal_kernel<float>(float *data, size_t size, float mean, float stddev,
                                                 unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  size_t vec_size = size / 4;
  size_t vec_stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < vec_size; i += vec_stride) {
    float4 r = curand_normal4(&state);
    float4 res;
    res.x = mean + stddev * r.x;
    res.y = mean + stddev * r.y;
    res.z = mean + stddev * r.z;
    res.w = mean + stddev * r.w;
    reinterpret_cast<float4 *>(data)[i] = res;
  }

  size_t remainder_start = vec_size * 4;
  if (idx == 0) {
    curandStatePhilox4_32_10_t state_rem;
    curand_init(seed, remainder_start, 0, &state_rem);
    for (size_t i = remainder_start; i < size; ++i) {
      data[i] = mean + stddev * curand_normal(&state_rem);
    }
  }
}

template <typename T, typename Func>
void dispatch_binary(const T *a, const T *b, T *c, size_t size, cudaStream_t stream, Func op) {
  if (size == 0)
    return;
  constexpr int vec_size = VectorizedTrait<T>::size;
  bool is_aligned =
      ((uintptr_t)a % 16 == 0) && ((uintptr_t)b % 16 == 0) && ((uintptr_t)c % 16 == 0);

  if (is_aligned && size % vec_size == 0) {
    int blocks = get_num_blocks(size / vec_size);
    binary_op_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size, op);
  } else {
    int blocks = get_num_blocks(size);
    binary_op_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size, op);
  }
  cuda::checkCudaError(cudaGetLastError(), "binary_op", __FILE__, __LINE__);
}

template <typename T, typename Func>
void dispatch_unary(const T *a, T *c, size_t size, cudaStream_t stream, Func op) {
  if (size == 0)
    return;
  constexpr int vec_size = VectorizedTrait<T>::size;
  bool is_aligned = ((uintptr_t)a % 16 == 0) && ((uintptr_t)c % 16 == 0);

  if (is_aligned && size % vec_size == 0) {
    int blocks = get_num_blocks(size / vec_size);
    unary_op_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, c, size, op);
  } else {
    int blocks = get_num_blocks(size);
    unary_op_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, c, size, op);
  }
  cuda::checkCudaError(cudaGetLastError(), "unary_op", __FILE__, __LINE__);
}

template <typename T, typename Func>
__global__ void ternary_op_kernel(const T *a, const T *b, T *c, size_t size, Func op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    c[idx] = op(a[idx], b[idx], c[idx]);
}

template <typename T, typename Func>
void dispatch_ternary(const T *a, const T *b, T *c, size_t size, cudaStream_t stream, Func op) {
  if (size == 0)
    return;
  int blocks = get_num_blocks(size);
  ternary_op_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size, op);
  cuda::checkCudaError(cudaGetLastError(), "ternary_op", __FILE__, __LINE__);
}

template <typename T, int Mode>
T dispatch_reduce(const T *a, const T *b, T scalar, size_t size, cudaStream_t stream) {
  if (size == 0)
    return (T)0;
  int blocks = std::min(get_num_blocks(size), 1024);

  T *d_partial;
  cudaMalloc(&d_partial, blocks * sizeof(T));
  reduce_kernel<T, Mode><<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, scalar, d_partial, size);

  T *h_partial = new T[blocks];
  cudaMemcpyAsync(h_partial, d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  T result = 0;
  for (int i = 0; i < blocks; ++i)
    result += h_partial[i];

  delete[] h_partial;
  cudaFree(d_partial);
  cuda::checkCudaError(cudaGetLastError(), "reduction", __FILE__, __LINE__);
  return result;
}

template <typename T>
void cuda_add(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Add<T>());
}
template <typename T>
void cuda_sub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Sub<T>());
}
template <typename T>
void cuda_mul(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Mul<T>());
}
template <typename T>
void cuda_div(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Div<T>());
}
template <typename T>
void cuda_min(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Min<T>());
}
template <typename T>
void cuda_max(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Max<T>());
}
template <typename T>
void cuda_equal(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Equal<T>());
}
template <typename T>
void cuda_greater(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_binary(a, b, c, size, stream, functors::Greater<T>());
}

template <typename T>
void cuda_add_scalar(const T *a, T s, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::AddScalar<T>{s});
}
template <typename T>
void cuda_sub_scalar(const T *a, T s, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::SubScalar<T>{s});
}
template <typename T>
void cuda_mul_scalar(const T *a, T s, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::MulScalar<T>{s});
}
template <typename T>
void cuda_div_scalar(const T *a, T s, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::DivScalar<T>{s});
}
template <typename T>
void cuda_scalar_max(const T *a, T s, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::ScalarMax<T>{s});
}
template <typename T>
void cuda_clamp(const T *a, T min, T max, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::Clamp<T>{min, max});
}

template <typename T>
void cuda_sub_mul_scalar(const T *a, T sub, T mul, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::SubMulScalar<T>{sub, mul});
}
template <typename T>
void cuda_mul_add_scalar(const T *a, T mul, T add, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::MulAddScalar<T>{mul, add});
}
template <typename T> void cuda_axpy(T alpha, const T *x, T *y, size_t size, cudaStream_t stream) {
  int blocks = get_num_blocks(size);
  axpy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(alpha, x, y, size);
  cuda::checkCudaError(cudaGetLastError(), "axpy", __FILE__, __LINE__);
}

template <typename T> void cuda_sqrt(const T *a, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::Sqrt<T>());
}
template <typename T> void cuda_abs(const T *a, T *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::Abs<T>());
}
void cuda_rsqrt(const float *a, float *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::Rsqrt<float>());
}
void cuda_rcp(const float *a, float *c, size_t size, cudaStream_t stream) {
  dispatch_unary(a, c, size, stream, functors::Rcp<float>());
}

template <typename T>
void cuda_fmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_ternary(a, b, c, size, stream, functors::FMAdd<T>());
}
template <typename T>
void cuda_fmsub(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_ternary(a, b, c, size, stream, functors::FMSub<T>());
}
template <typename T>
void cuda_fnmadd(const T *a, const T *b, T *c, size_t size, cudaStream_t stream) {
  dispatch_ternary(a, b, c, size, stream, functors::FNMAdd<T>());
}

template <typename T> void cuda_copy(const T *a, T *c, size_t size, cudaStream_t stream) {
  if (size == 0)
    return;
  cudaMemcpyAsync(c, a, size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  cuda::checkCudaError(cudaGetLastError(), "copy", __FILE__, __LINE__);
}

template <typename T> void cuda_set_scalar(T *c, T scalar, size_t size, cudaStream_t stream) {
  if (size == 0)
    return;
  int blocks = get_num_blocks(size);
  set_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(c, scalar, size);
  cuda::checkCudaError(cudaGetLastError(), "set_scalar", __FILE__, __LINE__);
}

template <typename T> void cuda_zero(T *c, size_t size, cudaStream_t stream) {
  if (size == 0)
    return;
  cudaMemsetAsync(c, 0, size * sizeof(T), stream);
  cuda::checkCudaError(cudaGetLastError(), "zero", __FILE__, __LINE__);
}

template <typename T>
void cuda_transpose_2d(const T *input, T *output, size_t rows, size_t cols, cudaStream_t stream) {
  if (rows == 0 || cols == 0)
    return;
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
  transpose_2d_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
  cuda::checkCudaError(cudaGetLastError(), "transpose_2d", __FILE__, __LINE__);
}

template <typename T>
void cuda_nchw_to_cnhw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream) {
  if (n == 0 || c == 0 || h == 0 || w == 0)
    return;
  dim3 grid((c + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);
  dim3 block(TILE_DIM, TILE_DIM);
  nchw_cnhw_transpose_tiled<<<grid, block, 0, stream>>>(input, output, n, c, h * w);
  cuda::checkCudaError(cudaGetLastError(), "nchw_to_cnhw", __FILE__, __LINE__);
}

template <typename T>
void cuda_cnhw_to_nchw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream) {
  cuda_nchw_to_cnhw(input, output, c, n, h, w, stream);
}

template <typename T>
void cuda_fill_random_uniform(T *data, size_t size, T min_val, T max_val, unsigned long long seed,
                              cudaStream_t stream) {
  if (size == 0)
    return;
  int blocks = get_num_blocks(size);
  fill_random_uniform_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, size, min_val, max_val, seed);
  cuda::checkCudaError(cudaGetLastError(), "fill_random_uniform", __FILE__, __LINE__);
}

template <typename T>
void cuda_fill_random_normal(T *data, size_t size, T mean, T stddev, unsigned long long seed,
                             cudaStream_t stream) {
  if (size == 0)
    return;
  int blocks = get_num_blocks(size);
  fill_random_normal_kernel<T><<<blocks, BLOCK_SIZE, 0, stream>>>(data, size, mean, stddev, seed);
  cuda::checkCudaError(cudaGetLastError(), "fill_random_normal", __FILE__, __LINE__);
}

template <typename T> T cuda_sum(const T *a, size_t size, cudaStream_t stream) {
  return dispatch_reduce<T, 0>(a, nullptr, (T)0, size, stream);
}

template <typename T> T cuda_dot_product(const T *a, const T *b, size_t size, cudaStream_t stream) {
  return dispatch_reduce<T, 1>(a, b, (T)0, size, stream);
}

template <typename T> T cuda_norm_squared(const T *a, size_t size, cudaStream_t stream) {
  return cuda_dot_product(a, a, size, stream);
}

template <typename T>
T cuda_sum_squared_diff(const T *a, T mean, size_t size, cudaStream_t stream) {
  return dispatch_reduce<T, 2>(a, nullptr, mean, size, stream);
}

#define INSTANTIATE_BIN(T)                                                                         \
  template void cuda_add<T>(const T *, const T *, T *, size_t, cudaStream_t);                      \
  template void cuda_sub<T>(const T *, const T *, T *, size_t, cudaStream_t);                      \
  template void cuda_mul<T>(const T *, const T *, T *, size_t, cudaStream_t);                      \
  template void cuda_div<T>(const T *, const T *, T *, size_t, cudaStream_t);                      \
  template void cuda_min<T>(const T *, const T *, T *, size_t, cudaStream_t);                      \
  template void cuda_max<T>(const T *, const T *, T *, size_t, cudaStream_t);                      \
  template void cuda_equal<T>(const T *, const T *, T *, size_t, cudaStream_t);                    \
  template void cuda_greater<T>(const T *, const T *, T *, size_t, cudaStream_t);                  \
  template void cuda_fmadd<T>(const T *, const T *, T *, size_t, cudaStream_t);                    \
  template void cuda_fmsub<T>(const T *, const T *, T *, size_t, cudaStream_t);                    \
  template void cuda_fnmadd<T>(const T *, const T *, T *, size_t, cudaStream_t);

#define INSTANTIATE_SCALAR(T)                                                                      \
  template void cuda_add_scalar<T>(const T *, T, T *, size_t, cudaStream_t);                       \
  template void cuda_sub_scalar<T>(const T *, T, T *, size_t, cudaStream_t);                       \
  template void cuda_mul_scalar<T>(const T *, T, T *, size_t, cudaStream_t);                       \
  template void cuda_div_scalar<T>(const T *, T, T *, size_t, cudaStream_t);                       \
  template void cuda_scalar_max<T>(const T *, T, T *, size_t, cudaStream_t);                       \
  template void cuda_clamp<T>(const T *, T, T, T *, size_t, cudaStream_t);                         \
  template void cuda_sub_mul_scalar<T>(const T *, T, T, T *, size_t, cudaStream_t);                \
  template void cuda_mul_add_scalar<T>(const T *, T, T, T *, size_t, cudaStream_t);                \
  template void cuda_axpy<T>(T, const T *, T *, size_t, cudaStream_t);

#define INSTANTIATE_UNARY(T)                                                                       \
  template void cuda_sqrt<T>(const T *, T *, size_t, cudaStream_t);                                \
  template void cuda_abs<T>(const T *, T *, size_t, cudaStream_t);

#define INSTANTIATE_UTILS(T)                                                                       \
  template void cuda_copy<T>(const T *, T *, size_t, cudaStream_t);                                \
  template void cuda_set_scalar<T>(T *, T, size_t, cudaStream_t);                                  \
  template void cuda_zero<T>(T *, size_t, cudaStream_t);                                           \
  template void cuda_transpose_2d<T>(const T *, T *, size_t, size_t, cudaStream_t);                \
  template void cuda_nchw_to_cnhw<T>(const T *, T *, size_t, size_t, size_t, size_t,               \
                                     cudaStream_t);                                                \
  template void cuda_cnhw_to_nchw<T>(const T *, T *, size_t, size_t, size_t, size_t,               \
                                     cudaStream_t);                                                \
  template void cuda_fill_random_uniform<T>(T *, size_t, T, T, unsigned long long, cudaStream_t);  \
  template void cuda_fill_random_normal<T>(T *, size_t, T, T, unsigned long long, cudaStream_t);   \
  template T cuda_sum<T>(const T *, size_t, cudaStream_t);                                         \
  template T cuda_dot_product<T>(const T *, const T *, size_t, cudaStream_t);                      \
  template T cuda_norm_squared<T>(const T *, size_t, cudaStream_t);                                \
  template T cuda_sum_squared_diff<T>(const T *, T, size_t, cudaStream_t);

INSTANTIATE_BIN(float)
INSTANTIATE_BIN(double)
INSTANTIATE_SCALAR(float)
INSTANTIATE_SCALAR(double)
INSTANTIATE_UNARY(float)
INSTANTIATE_UNARY(double)
INSTANTIATE_UTILS(float)
INSTANTIATE_UTILS(double)

} // namespace cuda
} // namespace tnn

#endif