/*
 * Copyright (c) 2025 Tung D. Pham
 */
#pragma once
#include <cstddef>

namespace tnn {
namespace cpu {

template <typename I_T, typename O_T>
void permute_heads(const I_T *input, O_T *output, size_t B, size_t L, size_t H, size_t D);

} // namespace cpu
} // namespace tnn
