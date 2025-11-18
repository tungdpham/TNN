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

__global__ void elu_gradient_kernel(const float *input, float *grad_output, size_t size,
                                    float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float local_grad = input[idx] > 0.0f ? 1.0f : alpha * expf(input[idx]);
    grad_output[idx] *= local_grad;
  }
}

template <>
void elu<float>(const float *input, float *output, size_t size, float alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient<float>(const float *input, float *grad_output, size_t size, float alpha,
                         cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, size, alpha);
}

} // namespace cuda
} // namespace tnn

#endif // USE_CUDA
