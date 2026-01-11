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

template <typename T>
void class_token_forward(const T *input, const T *token, T *output, size_t batch_size,
                         size_t seq_len, size_t embed_dim, cudaStream_t stream);

template <typename T>
void class_token_backward(const T *grad_output, T *grad_input, T *grad_token, size_t batch_size,
                          size_t seq_len, size_t embed_dim, cudaStream_t stream);
} // namespace cuda
} // namespace tnn
