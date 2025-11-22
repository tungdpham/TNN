#include "nn/activations_impl/cuda/tanh_kernels.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void tanh_kernel(const float *input, float *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanhf(input[idx]);
  }
}

__global__ void tanh_gradient_kernel(const float *input, float *grad_output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float tanh_val = tanhf(input[idx]);
    float local_grad = 1.0f - tanh_val * tanh_val;
    grad_output[idx] *= local_grad;
  }
}

__global__ void tanh_kernel_double(const double *input, double *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = ::tanh(input[idx]);
  }
}

__global__ void tanh_gradient_kernel_double(const double *input, double *grad_output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double tanh_val = ::tanh(input[idx]);
    double local_grad = 1.0 - tanh_val * tanh_val;
    grad_output[idx] *= local_grad;
  }
}

template <> void tanh<float>(const float *input, float *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void tanh_gradient<float>(const float *input, float *grad_output, size_t size,
                          cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size);
}

template <>
void tanh<double>(const double *input, double *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void tanh_gradient<double>(const double *input, double *grad_output, size_t size,
                           cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size);
}

} // namespace cuda
} // namespace tnn

#endif
