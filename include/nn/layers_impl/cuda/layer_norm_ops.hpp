/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace layer_norm {

template <typename T>
void layer_norm_forward(const T *input, T *output, const T *gamma, const T *beta, size_t batch_size,
                        size_t channels, size_t spatial_size, T epsilon, cudaStream_t stream);

template <typename T>
void layer_norm_backward(const T *grad_output, const T *input, const T *gamma, T *grad_input,
                         T *grad_gamma, T *grad_beta, size_t batch_size, size_t channels,
                         size_t spatial_size, T epsilon, cudaStream_t stream);

} // namespace layer_norm
} // namespace cuda
} // namespace tnn
