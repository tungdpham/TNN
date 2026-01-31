/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDNN
#include <cudnn.h>

namespace tnn {
namespace cuda {

template <typename T>
void softmax_forward(cudnnHandle_t handle, const T *input, T *output, size_t rows, size_t cols,
                     cudaStream_t stream);

template <typename T>
void softmax_backward(cudnnHandle_t handle, const T *output, const T *grad_output, T *grad_input,
                      size_t rows, size_t cols, cudaStream_t stream);

}  // namespace cuda
}  // namespace tnn
#endif
