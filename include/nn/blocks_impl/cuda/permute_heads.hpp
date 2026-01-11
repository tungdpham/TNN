/*
 * Copyright (c) 2025 Tung D. Pham
 */
#pragma once
#include <cstddef>

namespace tnn {
namespace cuda {

template <typename T>
void permute_heads(const T *input, T *output, size_t B, size_t L, size_t H, size_t D,
                   cudaStream_t stream);

} // namespace cuda
} // namespace tnn
