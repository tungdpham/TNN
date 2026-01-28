/*
 * Copyright (c) 2025 Tung D. Pham
 */
#include "nn/blocks_impl/cpu/permute_heads.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {

template <typename I_T, typename O_T>
void permute_heads(const I_T *input, O_T *output, size_t B, size_t L, size_t H, size_t D) {
  // Input: (B, L, H, D)
  // Output: (B, H, L, D)
  // Or vice versa if L and H are swapped in call.

  // Input strides for (B, L, H, D)
  size_t in_stride_b = L * H * D;
  size_t in_stride_l = H * D;
  size_t in_stride_h = D;

  // Output strides for (B, H, L, D) => output index for input[b,l,h,d]
  size_t out_stride_b = H * L * D;
  size_t out_stride_h = L * D;
  size_t out_stride_l = D;

  parallel_for_2d<size_t>(B, L, [&](size_t b, size_t l) {
    const I_T *src_base = input + b * in_stride_b + l * in_stride_l;
    for (size_t h = 0; h < H; ++h) {
      const I_T *src = src_base + h * in_stride_h;
      O_T *dst = output + b * out_stride_b + h * out_stride_h + l * out_stride_l;
      for (size_t d = 0; d < D; ++d) {
        dst[d] = static_cast<O_T>(src[d]);
      }
    }
  });
}

#define INSTANTIATE_PERMUTE_HEADS(I_T, O_T)                                                        \
  template void permute_heads<I_T, O_T>(const I_T *input, O_T *output, size_t B, size_t L,         \
                                        size_t H, size_t D);
#define INSTANTIATE_PERMUTE_HEADS_BOTH(I_T)                                                        \
  INSTANTIATE_PERMUTE_HEADS(I_T, fp16)                                                             \
  INSTANTIATE_PERMUTE_HEADS(I_T, bf16)                                                             \
  INSTANTIATE_PERMUTE_HEADS(I_T, float)                                                            \
  INSTANTIATE_PERMUTE_HEADS(I_T, double)

INSTANTIATE_PERMUTE_HEADS_BOTH(fp16)
INSTANTIATE_PERMUTE_HEADS_BOTH(bf16)
INSTANTIATE_PERMUTE_HEADS_BOTH(float)
INSTANTIATE_PERMUTE_HEADS_BOTH(double)

#undef INSTANTIATE_PERMUTE_HEADS_BOTH
#undef INSTANTIATE_PERMUTE_HEADS

} // namespace cpu
} // namespace tnn
