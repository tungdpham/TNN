#pragma once

#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
template <typename T>
void softmax(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
             size_t width, cudaStream_t stream);

template <typename T>
void softmax_gradient(const T *input, T *grad_output, size_t batch_size, size_t channels,
                      size_t height, size_t width, cudaStream_t stream);
} // namespace cuda
} // namespace tnn

#endif // USE_CUDA
