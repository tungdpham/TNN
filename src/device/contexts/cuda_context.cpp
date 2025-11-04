#include "device/contexts/cuda_context.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tdevice {

CUDAContext::CUDAContext(int id) : Context(id) {
  // Set the device for this context
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device " + std::to_string(id) + ": " +
                             cudaGetErrorString(err));
  }
}

void *CUDAContext::allocateMemory(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return ptr;
}

void CUDAContext::deallocateMemory(void *ptr) {
  if (ptr != nullptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      // Don't throw in destructor context, but could log the error
      // For now, we'll throw since this is a critical error
      throw std::runtime_error("Failed to free CUDA memory: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
}

void CUDAContext::copyToDevice(void *dest, const void *src, size_t size) {
  cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to copy memory to CUDA device: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void CUDAContext::copyToHost(void *dest, const void *src, size_t size) {
  cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to copy memory from CUDA device: " +
                             std::string(cudaGetErrorString(err)));
  }
}

} // namespace tdevice

#endif // USE_CUDA