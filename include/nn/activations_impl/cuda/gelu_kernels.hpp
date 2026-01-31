/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {

template <typename T>
void gelu(const T *input, T *output, size_t size, cudaStream_t stream);

template <typename T>
void gelu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size,
                   cudaStream_t stream);

}  // namespace cuda
}  // namespace tnn
