/*
 * Copyright (c) 2025 Tung D. Pham
 */
#include "nn/blocks_impl/cpu/causal_mask.hpp"

#include "type/type.hpp"

namespace tnn {
namespace cpu {

template <typename T>
void fill_causal_mask(T *mask, size_t batch_count, size_t L, T neg_inf) {
  for (size_t b = 0; b < batch_count; ++b) {
    for (size_t i = 0; i < L; ++i) {
      for (size_t j = 0; j < L; ++j) {
        mask[b * L * L + i * L + j] = (j > i) ? neg_inf : static_cast<T>(0);
      }
    }
  }
}

template <typename T>
void apply_causal_mask(T *scores, size_t batch_count, size_t L, T neg_inf) {
  for (size_t b = 0; b < batch_count; ++b) {
    // Each batch (head) has an LxL matrix
    T *batch_scores = scores + b * L * L;
    for (size_t i = 0; i < L; ++i) {
      for (size_t j = i + 1; j < L; ++j) {
        // Upper triangle where j > i
        batch_scores[i * L + j] = neg_inf;
      }
    }
  }
}

#define INSTANTIATE_APPLY_CAUSAL_MASK(T)                                                \
  template void fill_causal_mask<T>(T * mask, size_t batch_count, size_t L, T neg_inf); \
  template void apply_causal_mask<T>(T * scores, size_t batch_count, size_t L, T neg_inf);
INSTANTIATE_APPLY_CAUSAL_MASK(fp16)
INSTANTIATE_APPLY_CAUSAL_MASK(bf16)
INSTANTIATE_APPLY_CAUSAL_MASK(float)
INSTANTIATE_APPLY_CAUSAL_MASK(double)
#undef INSTANTIATE_APPLY_CAUSAL_MASK

}  // namespace cpu
}  // namespace tnn
