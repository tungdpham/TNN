#include "nn/activations_impl/cuda/elu_kernels.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void elu_kernel(const float *input, float *output, size_t size, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0f ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
  }
}

__global__ void elu_gradient_kernel(const float *input, const float *grad_output, float *grad_input,
                                    size_t size, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] =
        input[idx] > 0.0f ? grad_output[idx] : grad_output[idx] * alpha * expf(input[idx]);
  }
}

__global__ void elu_kernel_double(const double *input, double *output, size_t size, double alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0 ? input[idx] : alpha * (exp(input[idx]) - 1.0);
  }
}

__global__ void elu_gradient_kernel_double(const double *input, const double *grad_output,
                                           double *grad_input, size_t size, double alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] =
        input[idx] > 0.0 ? grad_output[idx] : grad_output[idx] * alpha * exp(input[idx]);
  }
}

template <>
void elu<float>(const float *input, float *output, size_t size, float alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                         size_t size, float alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input, size,
                                                            alpha);
}

template <>
void elu<double>(const double *input, double *output, size_t size, double alpha,
                 cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient<double>(const double *input, const double *grad_output, double *grad_input,
                          size_t size, double alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input,
                                                                   size, alpha);
}

} // namespace cuda
} // namespace tnn

#endif
