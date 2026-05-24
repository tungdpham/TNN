#include <cudnn_graph.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "device/cuda/cuda_context.hpp"
#include "device/flow.hpp"

namespace tnn {

CUDAContext::CUDAContext(int id)
    : Context(),
      device_id_(id) {
  // Set the device for this context
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device " + std::to_string(id) + ": " +
                             cudaGetErrorString(err));
  }
  nvmlInit_v2();
  createFlow(defaultFlowHandle);
}

#ifdef USE_CUDNN
cudnnHandle_t CUDAContext::getCudnnHandle() {
  cudnnHandle_t cudnn_handle = nullptr;
  cudnnStatus_t err = cudnnCreate(&cudnn_handle);
  if (err != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create CUDNN handle: " +
                             std::string(cudnnGetErrorString(err)));
  }
  return cudnn_handle;
}
#endif

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

size_t CUDAContext::getUsedMemory() const {
  nvmlDevice_t dev;
  nvmlReturn_t ret = nvmlDeviceGetHandleByIndex_v2(device_id_, &dev);
  if (ret != NVML_SUCCESS) {
    // Fall back to cudaMemGetInfo if NVML fails
    return getTotalMemory() - getAvailableMemory();
  }
  nvmlMemory_t mem;
  ret = nvmlDeviceGetMemoryInfo(dev, &mem);
  if (ret != NVML_SUCCESS) {
    return getTotalMemory() - getAvailableMemory();
  }
  return mem.used;
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
  (void)alignment;  // Unused parameter
  return allocateMemory(size);
}

void CUDAContext::deallocateAlignedMemory(void *ptr) { deallocateMemory(ptr); }

void CUDAContext::createFlow(flowHandle_t handle) {
  if (flows_.find(handle) == flows_.end()) {
    flows_[handle] = std::make_unique<CUDAFlow>();
  }
}

EngineType CUDAContext::get_engine() const { return EngineType::CUDA; }

Endianness CUDAContext::get_endianness() const {
  // CUDA devices are little-endian
  return Endianness::LITTLE;
}

Flow *CUDAContext::getFlow(flowHandle_t handle) {
  auto it = flows_.find(handle);
  if (it == flows_.end()) {
    std::cerr << "WARN: Creating new CUDAFlow with ID: " << handle
              << ". Are we using the right flow?" << std::endl;
    flows_[handle] = std::make_unique<CUDAFlow>();
    return flows_[handle].get();
  } else {
    return it->second.get();
  }
}

}  // namespace tnn

#endif  // USE_CUDA