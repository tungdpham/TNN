/*
 * Copyright (c) 2025 Tung D. Pham
 */
#include "nn/blocks_impl/cpu/causal_mask.hpp"

namespace tnn {
namespace cpu {

template <typename T> void fill_causal_mask(T *mask, size_t batch_count, size_t L, T neg_inf) {
  for (size_t b = 0; b < batch_count; ++b) {
    for (size_t i = 0; i < L; ++i) {
      for (size_t j = 0; j < L; ++j) {
        mask[b * L * L + i * L + j] = (j > i) ? neg_inf : static_cast<T>(0);
      }
    }
  }
}

template void fill_causal_mask<float>(float *mask, size_t batch_count, size_t L, float neg_inf);
template void fill_causal_mask<double>(double *mask, size_t batch_count, size_t L, double neg_inf);

} // namespace cpu
} // namespace tnn
