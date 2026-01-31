/*
 * Copyright (c) 2025 Tung D. Pham
 */
#pragma once
#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {

template <typename T>
void fill_causal_mask(T *mask, size_t batch_count, size_t L, T neg_inf, cudaStream_t stream = 0);

template <typename T>
void apply_causal_mask(T *scores, size_t batch_count, size_t L, T neg_inf, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace tnn
