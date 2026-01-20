#include "nn/activations_impl/cuda/leaky_relu_kernels.hpp"
#include "type/type.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void leaky_relu_kernel(const float *input, float *output, size_t size,
                                  float negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0f ? input[idx] : negative_slope * input[idx];
  }
}

__global__ void leaky_relu_gradient_kernel(const float *input, const float *grad_output,
                                           float *grad_input, size_t size, float negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : negative_slope * grad_output[idx];
  }
}

__global__ void leaky_relu_kernel_double(const double *input, double *output, size_t size,
                                         double negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0 ? input[idx] : negative_slope * input[idx];
  }
}

__global__ void leaky_relu_gradient_kernel_double(const double *input, const double *grad_output,
                                                  double *grad_input, size_t size,
                                                  double negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = input[idx] > 0.0 ? grad_output[idx] : negative_slope * grad_output[idx];
  }
}

template <>
void leaky_relu<float>(const float *input, float *output, size_t size, float negative_slope,
                       cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, negative_slope);
}

template <>
void leaky_relu_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                                size_t size, float negative_slope, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input,
                                                                   size, negative_slope);
}

template <>
void leaky_relu<double>(const double *input, double *output, size_t size, double negative_slope,
                        cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size,
                                                                 negative_slope);
}

template <>
void leaky_relu_gradient<double>(const double *input, const double *grad_output, double *grad_input,
                                 size_t size, double negative_slope, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
      input, grad_output, grad_input, size, negative_slope);
}

__global__ void leaky_relu_half_scalar_kernel(const fp16 *input, fp16 *output, size_t size,
                                              fp16 negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    fp16 zero = __float2half(0.0f);
    output[idx] = __hgt(input[idx], zero) ? input[idx] : __hmul(negative_slope, input[idx]);
  }
}

__global__ void leaky_relu_gradient_half_scalar_kernel(const fp16 *input, const fp16 *grad_output,
                                                       fp16 *grad_input, size_t size,
                                                       fp16 negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    fp16 zero = __float2half(0.0f);
    grad_input[idx] =
        __hgt(input[idx], zero) ? grad_output[idx] : __hmul(negative_slope, grad_output[idx]);
  }
}

template <>
void leaky_relu<fp16>(const fp16 *input, fp16 *output, size_t size, fp16 negative_slope,
                      cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_half_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size,
                                                                      negative_slope);
}

template <>
void leaky_relu_gradient<fp16>(const fp16 *input, const fp16 *grad_output, fp16 *grad_input,
                               size_t size, fp16 negative_slope, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_gradient_half_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
      input, grad_output, grad_input, size, negative_slope);
}

} // namespace cuda
} // namespace tnn

#endif
