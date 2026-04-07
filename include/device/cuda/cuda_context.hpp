/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDA

#include <cudnn_graph.h>

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "device/context.hpp"
#include "device/flow.hpp"

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tnn {
class CUDAContext : public Context {
private:
  std::unordered_map<flowHandle_t, std::unique_ptr<CUDAFlow>> flows_;
  int device_id_;

public:
  explicit CUDAContext(int id);
  static cudnnHandle_t getCudnnHandle();

  size_t getTotalMemory() const override;
  size_t getAvailableMemory() const override;
  size_t getUsedMemory() const override;
  void *allocateMemory(size_t size) override;
  void deallocateMemory(void *ptr) override;
  void *allocateAlignedMemory(size_t size, size_t alignment) override;
  void deallocateAlignedMemory(void *ptr) override;
  void copyToDevice(void *dest, const void *src, size_t size) override;
  void copyToHost(void *dest, const void *src, size_t size) override;
  EngineType get_engine() const override;
  void createFlow(flowHandle_t handle) override;
  Endianness get_endianness() const override;
  Flow *getFlow(flowHandle_t handle) override;
};
}  // namespace tnn

#endif  // USE_CUDA