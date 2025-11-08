#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda {
inline void checkCudaError(cudaError_t result, const char *func, const char *file, int line) {
  if (result != cudaSuccess) {
    std::string errorMessage = "CUDA Error: " + std::string(cudaGetErrorString(result)) + " at " +
                               std::string(file) + ":" + std::to_string(line) + " in function " +
                               std::string(func);
    throw std::runtime_error(errorMessage);
  }
}
} // namespace cuda

#define CUDA_CHECK(call) cuda::checkCudaError((call), __func__, __FILE__, __LINE__)
#endif // USE_CUDA