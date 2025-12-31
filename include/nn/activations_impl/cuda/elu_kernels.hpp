#pragma once

#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
template <typename T>
void elu(const T *input, T *output, size_t size, T alpha, cudaStream_t stream);

template <typename T>
void elu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size, T alpha,
                  cudaStream_t stream);
} // namespace cuda
} // namespace tnn

#endif // USE_CUDA
