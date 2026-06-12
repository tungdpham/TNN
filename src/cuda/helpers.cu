#include <cuda_runtime.h>

#include "cuda/helpers.hpp"

namespace synet {
namespace cuda {
void synchronize() { cudaDeviceSynchronize(); }
}  // namespace cuda
}  // namespace synet