#pragma once

#ifdef USE_CUDA

#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, T *mask_data, size_t batch_size,
                             size_t channels, size_t spatial_size, T dropout_rate,
                             cudaStream_t stream);
}
} // namespace tnn

#endif