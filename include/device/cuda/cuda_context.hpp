#pragma once

#ifdef USE_CUDA

#include "device/context.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace tnn {
class CUDAContext : public Context {
private:
  std::unordered_map<std::string, std::unique_ptr<Flow>> flows_;

public:
  explicit CUDAContext(int id);

  size_t getTotalMemory() const override;
  size_t getAvailableMemory() const override;
  void *allocateMemory(size_t size) override;
  void deallocateMemory(void *ptr) override;
  void *allocateAlignedMemory(size_t size, size_t alignment) override;
  void deallocateAlignedMemory(void *ptr) override;
  void copyToDevice(void *dest, const void *src, size_t size) override;
  void copyToHost(void *dest, const void *src, size_t size) override;
  void createFlow(const std::string &flow_id) override;
  Flow *getFlow(const std::string &flow_id) override;
};
} // namespace tnn

#endif // USE_CUDA