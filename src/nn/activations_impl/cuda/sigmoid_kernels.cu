#include "nn/activations_impl/cuda/sigmoid_kernels.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void sigmoid_kernel(const float *input, float *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

__global__ void sigmoid_gradient_kernel(const float *input, const float *grad_output,
                                        float *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sigmoid_val = 1.0f / (1.0f + expf(-input[idx]));
    grad_input[idx] = grad_output[idx] * sigmoid_val * (1.0f - sigmoid_val);
  }
}

__global__ void sigmoid_kernel_double(const double *input, double *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
  }
}

__global__ void sigmoid_gradient_kernel_double(const double *input, const double *grad_output,
                                               double *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double sigmoid_val = 1.0 / (1.0 + exp(-input[idx]));
    grad_input[idx] = grad_output[idx] * sigmoid_val * (1.0 - sigmoid_val);
  }
}

template <>
void sigmoid<float>(const float *input, float *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void sigmoid_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                             size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input,
                                                                size);
}

template <>
void sigmoid<double>(const double *input, double *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void sigmoid_gradient<double>(const double *input, const double *grad_output, double *grad_input,
                              size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                       grad_input, size);
}

} // namespace cuda
} // namespace tnn

#endif
