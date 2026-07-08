#include "type/type.hpp"

#ifdef USE_CUDA
#include <cuda_bf16.h>
#endif

namespace tunx {
const float TypeTraits<int8>::epsilon = 1e-1f;
const float TypeTraits<fp16>::epsilon = 1e-3f;
const float TypeTraits<bf16>::epsilon = 1e-2f;
const float TypeTraits<fp32>::epsilon = 1e-5f;
const float TypeTraits<fp64>::epsilon = 1e-8f;
}  // namespace tunx