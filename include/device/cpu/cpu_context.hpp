/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "device/context.hpp"

namespace tnn {
class CPUContext : public Context {
  std::unordered_map<flowHandle_t, std::unique_ptr<CPUFlow>> flows_;

public:
  explicit CPUContext();

  size_t getTotalMemory() const override;
  size_t getAvailableMemory() const override;
  void *allocateMemory(size_t size) override;
  void deallocateMemory(void *ptr) override;
  void *allocateAlignedMemory(size_t size, size_t alignment) override;
  void deallocateAlignedMemory(void *ptr) override;
  void copyToDevice(void *dest, const void *src, size_t size) override;
  void copyToHost(void *dest, const void *src, size_t size) override;
  void createFlow(flowHandle_t handle) override;
  EngineType get_engine() const override;
  Endianness get_endianness() const override;
  Flow *getFlow(flowHandle_t handle) override;
};
}  // namespace tnn