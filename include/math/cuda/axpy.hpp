#pragma once
#include <cuda_runtime.h>

#include "type/type.hpp"

namespace tunx {
namespace cuda {

void axpy(const void* X, void* Y, size_t num_elements, DType_t dtype, cudaStream_t stream);

}
}  // namespace tunx
