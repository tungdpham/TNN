#include "nn/activations_impl/cuda/relu_kernels.hpp"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef USE_CUDA

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void relu_vec4_kernel(const float *__restrict__ input, float *__restrict__ output,
                                 size_t size) {

  size_t vec_size = size / 4;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const float4 *in_vec = reinterpret_cast<const float4 *>(input);
  float4 *out_vec = reinterpret_cast<float4 *>(output);

  for (size_t i = idx; i < vec_size; i += stride) {
    float4 v = in_vec[i];
    v.x = fmaxf(0.0f, v.x);
    v.y = fmaxf(0.0f, v.y);
    v.z = fmaxf(0.0f, v.z);
    v.w = fmaxf(0.0f, v.w);
    out_vec[i] = v;
  }

  size_t tail_start = vec_size * 4;
  for (size_t i = tail_start + idx; i < size; i += stride) {
    output[i] = fmaxf(0.0f, input[i]);
  }
}

__global__ void relu_gradient_vec4_kernel(const float *__restrict__ input,
                                          float *__restrict__ grad_output, size_t size) {
  size_t vec_size = size / 4;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const float4 *in_vec = reinterpret_cast<const float4 *>(input);
  float4 *grad_vec = reinterpret_cast<float4 *>(grad_output);

  for (size_t i = idx; i < vec_size; i += stride) {
    float4 in_v = in_vec[i];
    float4 grad_v = grad_vec[i];

    grad_v.x = (in_v.x > 0.0f) ? grad_v.x : 0.0f;
    grad_v.y = (in_v.y > 0.0f) ? grad_v.y : 0.0f;
    grad_v.z = (in_v.z > 0.0f) ? grad_v.z : 0.0f;
    grad_v.w = (in_v.w > 0.0f) ? grad_v.w : 0.0f;

    grad_vec[i] = grad_v;
  }

  size_t tail_start = vec_size * 4;
  for (size_t i = tail_start + idx; i < size; i += stride) {
    grad_output[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
  }
}

__global__ void relu_vec2_double_kernel(const double *__restrict__ input,
                                        double *__restrict__ output, size_t size) {
  size_t vec_size = size / 2;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const double2 *in_vec = reinterpret_cast<const double2 *>(input);
  double2 *out_vec = reinterpret_cast<double2 *>(output);

  for (size_t i = idx; i < vec_size; i += stride) {
    double2 v = in_vec[i];
    v.x = fmax(0.0, v.x);
    v.y = fmax(0.0, v.y);
    out_vec[i] = v;
  }

  size_t tail_start = vec_size * 2;
  for (size_t i = tail_start + idx; i < size; i += stride) {
    output[i] = fmax(0.0, input[i]);
  }
}

__global__ void relu_gradient_vec2_double_kernel(const double *__restrict__ input,
                                                 double *__restrict__ grad_output, size_t size) {
  size_t vec_size = size / 2;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const double2 *in_vec = reinterpret_cast<const double2 *>(input);
  double2 *grad_vec = reinterpret_cast<double2 *>(grad_output);

  for (size_t i = idx; i < vec_size; i += stride) {
    double2 in_v = in_vec[i];
    double2 grad_v = grad_vec[i];

    grad_v.x = (in_v.x > 0.0) ? grad_v.x : 0.0;
    grad_v.y = (in_v.y > 0.0) ? grad_v.y : 0.0;

    grad_vec[i] = grad_v;
  }

  size_t tail_start = vec_size * 2;
  for (size_t i = tail_start + idx; i < size; i += stride) {
    grad_output[i] = (input[i] > 0.0) ? grad_output[i] : 0.0;
  }
}

__global__ void relu_scalar_kernel(const float *input, float *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < size; i += stride) {
    output[i] = fmaxf(0.0f, input[i]);
  }
}

__global__ void relu_gradient_scalar_kernel(const float *input, float *grad_output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < size; i += stride) {
    grad_output[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
  }
}

__global__ void relu_double_scalar_kernel(const double *input, double *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < size; i += stride) {
    output[i] = fmax(0.0, input[i]);
  }
}

__global__ void relu_gradient_double_scalar_kernel(const double *input, double *grad_output,
                                                   size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < size; i += stride) {
    grad_output[i] = (input[i] > 0.0) ? grad_output[i] : 0.0;
  }
}

template <> void relu<float>(const float *input, float *output, size_t size, cudaStream_t stream) {

  bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(output) % 16 == 0);

  if (is_aligned) {
    size_t vec_size = size / 4;
    int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks == 0)
      num_blocks = 1;
    relu_vec4_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
  } else {
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
  }
}

template <>
void relu_gradient<float>(const float *input, float *grad_output, size_t size,
                          cudaStream_t stream) {
  bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(grad_output) % 16 == 0);

  if (is_aligned) {
    size_t vec_size = size / 4;
    int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0)
      num_blocks = 1;
    relu_gradient_vec4_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size);
  } else {
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_gradient_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size);
  }
}

template <>
void relu<double>(const double *input, double *output, size_t size, cudaStream_t stream) {
  bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(output) % 16 == 0);

  if (is_aligned) {
    size_t vec_size = size / 2;
    int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0)
      num_blocks = 1;
    relu_vec2_double_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
  } else {
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_double_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
  }
}

template <>
void relu_gradient<double>(const double *input, double *grad_output, size_t size,
                           cudaStream_t stream) {
  bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(grad_output) % 16 == 0);

  if (is_aligned) {
    size_t vec_size = size / 2;
    int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0)
      num_blocks = 1;
    relu_gradient_vec2_double_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                            size);
  } else {
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_gradient_double_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                              size);
  }
}

} // namespace cuda
} // namespace tnn

#endif