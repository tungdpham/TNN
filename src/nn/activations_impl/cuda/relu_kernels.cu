#include "nn/activations_impl/cuda/relu_kernels.hpp"
#include "type/type.hpp"
#include <cmath>
#include <cuda_fp16.h>
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
                                          const float *__restrict__ grad_output,
                                          float *__restrict__ grad_input, size_t size) {
  size_t vec_size = size / 4;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const float4 *in_vec = reinterpret_cast<const float4 *>(input);
  const float4 *grad_out_vec = reinterpret_cast<const float4 *>(grad_output);
  float4 *grad_in_vec = reinterpret_cast<float4 *>(grad_input);

  for (size_t i = idx; i < vec_size; i += stride) {
    float4 in_v = in_vec[i];
    float4 grad_out_v = grad_out_vec[i];
    float4 grad_in_v;

    grad_in_v.x = (in_v.x > 0.0f) ? grad_out_v.x : 0.0f;
    grad_in_v.y = (in_v.y > 0.0f) ? grad_out_v.y : 0.0f;
    grad_in_v.z = (in_v.z > 0.0f) ? grad_out_v.z : 0.0f;
    grad_in_v.w = (in_v.w > 0.0f) ? grad_out_v.w : 0.0f;

    grad_in_vec[i] = grad_in_v;
  }

  size_t tail_start = vec_size * 4;
  for (size_t i = tail_start + idx; i < size; i += stride) {
    grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
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
                                                 const double *__restrict__ grad_output,
                                                 double *__restrict__ grad_input, size_t size) {
  size_t vec_size = size / 2;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const double2 *in_vec = reinterpret_cast<const double2 *>(input);
  const double2 *grad_out_vec = reinterpret_cast<const double2 *>(grad_output);
  double2 *grad_in_vec = reinterpret_cast<double2 *>(grad_input);

  for (size_t i = idx; i < vec_size; i += stride) {
    double2 in_v = in_vec[i];
    double2 grad_out_v = grad_out_vec[i];
    double2 grad_in_v;

    grad_in_v.x = (in_v.x > 0.0) ? grad_out_v.x : 0.0;
    grad_in_v.y = (in_v.y > 0.0) ? grad_out_v.y : 0.0;

    grad_in_vec[i] = grad_in_v;
  }

  size_t tail_start = vec_size * 2;
  for (size_t i = tail_start + idx; i < size; i += stride) {
    grad_input[i] = (input[i] > 0.0) ? grad_output[i] : 0.0;
  }
}

__global__ void relu_half_scalar_kernel(const fp16 *input, fp16 *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < size; i += stride) {
    output[i] = __hgt(input[i], __float2half(0.0f)) ? input[i] : __float2half(0.0f);
  }
}

__global__ void relu_gradient_half_scalar_kernel(const fp16 *input, const fp16 *grad_output,
                                                 fp16 *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < size; i += stride) {
    grad_input[i] = __hgt(input[i], __float2half(0.0f)) ? grad_output[i] : __float2half(0.0f);
  }
}

template <> void relu<float>(const float *input, float *output, size_t size, cudaStream_t stream) {
  size_t vec_size = size / 4;
  int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (num_blocks == 0)
    num_blocks = 1;
  relu_vec4_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void relu_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                          size_t size, cudaStream_t stream) {
  size_t vec_size = size / 4;
  int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_blocks == 0)
    num_blocks = 1;
  relu_gradient_vec4_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input,
                                                                   size);
}

template <>
void relu<double>(const double *input, double *output, size_t size, cudaStream_t stream) {
  size_t vec_size = size / 2;
  int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_blocks == 0)
    num_blocks = 1;
  relu_vec2_double_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void relu_gradient<double>(const double *input, const double *grad_output, double *grad_input,
                           size_t size, cudaStream_t stream) {
  size_t vec_size = size / 2;
  int num_blocks = (vec_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_blocks == 0)
    num_blocks = 1;
  relu_gradient_vec2_double_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                          grad_input, size);
}

template <> void relu<fp16>(const fp16 *input, fp16 *output, size_t size, cudaStream_t stream) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_blocks == 0)
    num_blocks = 1;
  relu_half_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void relu_gradient<fp16>(const fp16 *input, const fp16 *grad_output, fp16 *grad_input, size_t size,
                         cudaStream_t stream) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_blocks == 0)
    num_blocks = 1;
  relu_gradient_half_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                          grad_input, size);
}

} // namespace cuda
} // namespace tnn

#endif