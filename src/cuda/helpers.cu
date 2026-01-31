#include <cuda_runtime.h>

#include "cuda/helpers.hpp"

namespace tnn {
namespace cuda {
void synchronize() { cudaDeviceSynchronize(); }
}  // namespace cuda
}  // namespace tnn