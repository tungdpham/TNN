#include "cuda/helpers.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
void synchronize() { cudaDeviceSynchronize(); }
} // namespace cuda
} // namespace tnn