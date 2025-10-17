/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"
#include <string>

namespace tpipeline {
enum TaskType { FORWARD, BACKWARD };

template <typename T = float> struct Task {
  Tensor<T> data;
  size_t micro_batch_id;

  Task() = default;

  Task(const Tensor<T> &d, size_t mb_id) : data(d.clone()), micro_batch_id(mb_id) {}

  Task(const Task &other) : data(other.data.clone()), micro_batch_id(other.micro_batch_id) {}

  Task(Task &&other) noexcept : data(std::move(other.data)), micro_batch_id(other.micro_batch_id) {}

  Task &operator=(const Task &other) {
    if (this != &other) {
      data = other.data.clone();
      micro_batch_id = other.micro_batch_id;
    }
    return *this;
  }

  Task &operator=(Task &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
      micro_batch_id = other.micro_batch_id;
    }
    return *this;
  }
};
} // namespace tpipeline