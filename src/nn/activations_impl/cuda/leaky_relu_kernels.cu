#include "nn/activations_impl/cuda/leaky_relu_kernels.hpp"

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

__global__ void leaky_relu_gradient_kernel(const float *input, float *grad_output, size_t size,
                                           float negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float local_grad = input[idx] > 0.0f ? 1.0f : negative_slope;
    grad_output[idx] *= local_grad;
  }
}

__global__ void leaky_relu_kernel_double(const double *input, double *output, size_t size,
                                         double negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0 ? input[idx] : negative_slope * input[idx];
  }
}

__global__ void leaky_relu_gradient_kernel_double(const double *input, double *grad_output,
                                                  size_t size, double negative_slope) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double local_grad = input[idx] > 0.0 ? 1.0 : negative_slope;
    grad_output[idx] *= local_grad;
  }
}

template <>
void leaky_relu<float>(const float *input, float *output, size_t size, float negative_slope,
                       cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, negative_slope);
}

template <>
void leaky_relu_gradient<float>(const float *input, float *grad_output, size_t size,
                                float negative_slope, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size,
                                                                   negative_slope);
}

template <>
void leaky_relu<double>(const double *input, double *output, size_t size, double negative_slope,
                        cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size,
                                                                 negative_slope);
}

template <>
void leaky_relu_gradient<double>(const double *input, double *grad_output, size_t size,
                                 double negative_slope, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  leaky_relu_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size,
                                                                          negative_slope);
}

} // namespace cuda
} // namespace tnn

#endif
