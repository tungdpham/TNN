#pragma once

#include <cudnn_graph.h>

#include "device/flow.hpp"
#ifdef USE_CUDA

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "device/context.hpp"

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tnn {
class CUDAContext : public Context {
private:
  std::unordered_map<flowHandle_t, std::unique_ptr<CUDAFlow>> flows_;
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle_;
#endif

public:
  explicit CUDAContext(int id);
#ifdef USE_CUDNN
  ~CUDAContext();
  cudnnHandle_t getCudnnHandle() const { return cudnn_handle_; }
#endif

  size_t getTotalMemory() const override;
  size_t getAvailableMemory() const override;
  void *allocateMemory(size_t size) override;
  void deallocateMemory(void *ptr) override;
  void *allocateAlignedMemory(size_t size, size_t alignment) override;
  void deallocateAlignedMemory(void *ptr) override;
  void copyToDevice(void *dest, const void *src, size_t size) override;
  void copyToHost(void *dest, const void *src, size_t size) override;
  void createFlow(flowHandle_t handle) override;
  Flow *getFlow(flowHandle_t handle) override;
};
}  // namespace tnn

#endif  // USE_CUDA