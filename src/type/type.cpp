#include "type/type.hpp"

#ifdef USE_CUDA
#include <cuda_bf16.h>
#endif

namespace tnn {
#ifdef USE_CUDA
const float TypeTraits<fp16>::epsilon = 1e-3f;
const float TypeTraits<fp32>::epsilon = 1e-5f;
const float TypeTraits<fp64>::epsilon = 1e-8f;
#else
#endif
} // namespace tnn