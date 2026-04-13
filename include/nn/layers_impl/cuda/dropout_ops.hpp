#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {
namespace dropout {

template <typename T>
void run_forward(const T *input_data, T *output_data, bool *mask_data, size_t batch_size,
                 size_t channels, size_t spatial_size, T dropout_rate, cudaStream_t stream);

template <typename T>
void run_backward(const T *grad_output_data, T *grad_input_data, const bool *mask_data,
                  size_t batch_size, size_t channels, size_t spatial_size, T scale,
                  cudaStream_t stream);

}  // namespace dropout
}  // namespace cuda
}  // namespace tnn

#endif