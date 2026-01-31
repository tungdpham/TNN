#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {
template <typename T>
void relu(const T *input, T *output, size_t size, cudaStream_t stream);

template <typename T>
void relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size,
                   cudaStream_t stream);
}  // namespace cuda
}  // namespace tnn

#endif  // USE_CUDA
