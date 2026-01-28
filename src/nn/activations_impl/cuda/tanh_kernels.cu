#include "nn/activations_impl/cuda/tanh_kernels.hpp"
#include "type/type.hpp"

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

__global__ void tanh_gradient_kernel(const float *input, const float *grad_output,
                                     float *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float tanh_val = tanhf(input[idx]);
    grad_input[idx] = grad_output[idx] * (1.0f - tanh_val * tanh_val);
  }
}

__global__ void tanh_kernel_double(const double *input, double *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = ::tanh(input[idx]);
  }
}

__global__ void tanh_gradient_kernel_double(const double *input, const double *grad_output,
                                            double *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double tanh_val = ::tanh(input[idx]);
    grad_input[idx] = grad_output[idx] * (1.0 - tanh_val * tanh_val);
  }
}

template <> void tanh<float>(const float *input, float *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void tanh_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                          size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input, size);
}

template <>
void tanh<double>(const double *input, double *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void tanh_gradient<double>(const double *input, const double *grad_output, double *grad_input,
                           size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input,
                                                                    size);
}

__global__ void tanh_half_scalar_kernel(const fp16 *input, fp16 *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float input_f = __half2float(input[idx]);
    output[idx] = __float2half(tanhf(input_f));
  }
}

__global__ void tanh_gradient_half_scalar_kernel(const fp16 *input, const fp16 *grad_output,
                                                 fp16 *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float input_f = __half2float(input[idx]);
    float tanh_val_f = tanhf(input_f);
    fp16 tanh_val = __float2half(tanh_val_f);
    fp16 one = __float2half(1.0f);
    fp16 grad = __hsub(one, __hmul(tanh_val, tanh_val));
    grad_input[idx] = __hmul(grad_output[idx], grad);
  }
}

template <> void tanh<fp16>(const fp16 *input, fp16 *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_half_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void tanh_gradient<fp16>(const fp16 *input, const fp16 *grad_output, fp16 *grad_input, size_t size,
                         cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_gradient_half_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                         grad_input, size);
}

__global__ void tanh_bf16_scalar_kernel(const bf16 *input, bf16 *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float input_f = static_cast<float>(input[idx]);
    output[idx] = __float2bfloat16(tanhf(input_f));
  }
}

__global__ void tanh_gradient_bf16_scalar_kernel(const bf16 *input, const bf16 *grad_output,
                                                 bf16 *grad_input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float input_f = static_cast<float>(input[idx]);
    float tanh_val_f = tanhf(input_f);
    bf16 tanh_val = __float2bfloat16(tanh_val_f);
    bf16 one = __float2bfloat16(1.0f);
    bf16 grad = one - tanh_val * tanh_val;
    grad_input[idx] = grad_output[idx] * grad;
  }
}

template <> void tanh<bf16>(const bf16 *input, bf16 *output, size_t size, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_bf16_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size);
}

template <>
void tanh_gradient<bf16>(const bf16 *input, const bf16 *grad_output, bf16 *grad_input, size_t size,
                         cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_gradient_bf16_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                         grad_input, size);
}

} // namespace cuda
} // namespace tnn

#endif
