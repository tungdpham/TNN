/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "device/ibv_allocator.hpp"
#include "device/sref.hpp"
#include "distributed/ibuffer.hpp"

namespace tnn {

class RoCEBuffer : public IBuffer {
private:
  sref<IbvAllocator> allocator_;

  bool in_range(size_t index) const { return index < size_; }

  std::string get_out_of_bound_msg(int index) const {
    return "Array bounds is (0, " + std::to_string(size_) + "), accessed: " + std::to_string(index);
  }

public:
  using IBuffer::IBuffer;

  RoCEBuffer(IbvAllocator &allocator, size_t capacity)
      : IBuffer(allocator.allocate(capacity)),
        allocator_(allocator) {}

  RoCEBuffer(const RoCEBuffer &) = delete;
  RoCEBuffer &operator=(const RoCEBuffer &) = delete;
  RoCEBuffer(RoCEBuffer &&other) noexcept
      : IBuffer(std::move(other)),
        allocator_(other.allocator_) {}

  RoCEBuffer &operator=(RoCEBuffer &&other) noexcept {
    if (this != &other) {
      IBuffer::operator=(std::move(other));
      allocator_ = other.allocator_;
    }
    return *this;
  }

  ibv_mr *get_mr() const { return allocator_->get_mr(); }
  uint32_t get_lkey() const { return allocator_->get_mr_info(this->data_).lkey; }
  uint32_t get_rkey() const { return allocator_->get_mr_info(this->data_).rkey; }
};

}  // namespace tnn
