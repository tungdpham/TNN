/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/cuda/gelu_kernels.hpp"
#include "type/type.hpp"
#include <cmath>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {

template <typename T> __global__ void gelu_kernel(const T *input, T *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T x = input[idx];
    const T kAlpha = static_cast<T>(0.79788456080286535587989211986876);
    const T kBeta = static_cast<T>(0.044715);
    T inner = kAlpha * (x + kBeta * x * x * x);
    output[idx] = static_cast<T>(0.5) * x *
                  (static_cast<T>(1.0) + static_cast<T>(tanh(static_cast<double>(inner))));
  }
}

template <typename T> void gelu(const T *input, T *output, size_t size, cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  gelu_kernel<T><<<blocks, threads, 0, stream>>>(input, output, size);
}

template <typename T>
__global__ void gelu_gradient_kernel(const T *input, const T *grad_output, T *grad_input,
                                     size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T x = input[idx];
    T dy = grad_output[idx];
    const T kAlpha = static_cast<T>(0.79788456080286535587989211986876);
    const T kBeta = static_cast<T>(0.044715);

    T x3 = x * x * x;
    T y = kAlpha * (x + kBeta * x3);
    T tanh_y = static_cast<T>(tanh(static_cast<double>(y)));

    T sech_sq = (static_cast<T>(1.0) - tanh_y * tanh_y);
    T dy_dx = kAlpha * (static_cast<T>(1.0) + static_cast<T>(3.0) * kBeta * x * x);

    T dGELU = static_cast<T>(0.5) * (static_cast<T>(1.0) + tanh_y) +
              static_cast<T>(0.5) * x * sech_sq * dy_dx;

    grad_input[idx] = dy * dGELU;
  }
}

template <typename T>
void gelu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size,
                   cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  gelu_gradient_kernel<T><<<blocks, threads, 0, stream>>>(input, grad_output, grad_input, size);
}

#define INSTANTIATE(T)                                                                             \
  template void gelu<T>(const T *input, T *output, size_t size, cudaStream_t stream);              \
  template void gelu_gradient<T>(const T *input, const T *grad_output, T *grad_input, size_t size, \
                                 cudaStream_t stream);

INSTANTIATE(fp16);
INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE
} // namespace cuda
} // namespace tnn
