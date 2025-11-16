#include "device/cuda/cuda_context.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tnn {

CUDAContext::CUDAContext(int id) : Context() {
  // Set the device for this context
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device " + std::to_string(id) + ": " +
                             cudaGetErrorString(err));
  }
}

size_t CUDAContext::getTotalMemory() const {
  size_t total_mem = 0;
  cudaError_t err = cudaMemGetInfo(nullptr, &total_mem);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get total CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return total_mem;
}

size_t CUDAContext::getAvailableMemory() const {
  size_t free_mem = 0;
  cudaError_t err = cudaMemGetInfo(&free_mem, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get available CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return free_mem;
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

void *CUDAContext::allocateAlignedMemory(size_t size, size_t alignment) {
  // cudaMalloc already provides 256-byte alignment, which is sufficient for most cases
  (void)alignment; // Unused parameter
  return allocateMemory(size);
}
void CUDAContext::deallocateAlignedMemory(void *ptr) { deallocateMemory(ptr); }

} // namespace tnn

#endif // USE_CUDA