/*
 * Copyright (c) 2025 Tung D. Pham
 */
#pragma once
#include <cstddef>

namespace tnn {
namespace cpu {

template <typename T> void fill_causal_mask(T *mask, size_t batch_count, size_t L, T neg_inf);

} // namespace cpu
} // namespace tnn
