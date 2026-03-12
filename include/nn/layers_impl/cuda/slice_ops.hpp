/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <vector>

namespace tnn {
namespace cuda {
namespace slice {

template <typename T>
void slice_forward(const T *input, T *output, const std::vector<size_t> &input_shape, size_t axis,
                   size_t start, size_t length, cudaStream_t stream);

template <typename T>
void slice_backward(const T *grad_output, T *grad_input, const std::vector<size_t> &input_shape,
                    size_t axis, size_t start, size_t length, cudaStream_t stream);

}  // namespace slice
}  // namespace cuda
}  // namespace tnn

#endif  // USE_CUDA
