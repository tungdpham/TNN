#include "ops/cpu/permute_heads.hpp"

#include "type/type.hpp"

namespace tunx {
namespace ops {
namespace cpu {

template <typename I_T, typename O_T>
void permute_heads(const I_T *input, O_T *output, size_t B, size_t L, size_t H, size_t D) {
  for (size_t b = 0; b < B; ++b) {
    for (size_t l = 0; l < L; ++l) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t d = 0; d < D; ++d) {
          size_t in_idx = b * (L * H * D) + l * (H * D) + h * D + d;
          size_t out_idx = b * (H * L * D) + h * (L * D) + l * D + d;
          output[out_idx] = static_cast<O_T>(input[in_idx]);
        }
      }
    }
  }
}

#define INSTANTIATE_BOTH(I_T, O_T)                                                                \
  template void permute_heads<I_T, O_T>(const I_T *input, O_T *output, size_t B, size_t L,        \
                                        size_t H, size_t D);
#define INSTANTIATE(I_T)       \
  INSTANTIATE_BOTH(I_T, int8)  \
  INSTANTIATE_BOTH(I_T, fp16)  \
  INSTANTIATE_BOTH(I_T, bf16)  \
  INSTANTIATE_BOTH(I_T, float) \
  INSTANTIATE_BOTH(I_T, double)

#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE
#undef INSTANTIATE_BOTH

}  // namespace cpu
}  // namespace ops
}  // namespace tunx
