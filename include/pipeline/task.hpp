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
  TaskType type;
  Tensor<T> data;
  size_t micro_batch_id;

  Task() = default;
  Task(TaskType t, const Tensor<T> &d, size_t mb_id)
      : type(t), data(d), micro_batch_id(mb_id) {}

  Task(const Task &other)
      : type(other.type), data(other.data.clone()),
        micro_batch_id(other.micro_batch_id) {}

  Task &operator=(const Task &other) {
    if (this != &other) {
      type = other.type;
      data = other.data.clone();
      micro_batch_id = other.micro_batch_id;
    }
    return *this;
  }

  Task(Task &&other) noexcept
      : type(other.type), data(std::move(other.data)),
        micro_batch_id(other.micro_batch_id) {}

  Task &operator=(Task &&other) noexcept {
    if (this != &other) {
      type = other.type;
      data = std::move(other.data);
      micro_batch_id = other.micro_batch_id;
    }
    return *this;
  }

  std::string to_string() const {
    return "Task(type: " + std::to_string(static_cast<int>(type)) +
           ", micro_batch_id: " + std::to_string(micro_batch_id) +
           ", data_shape: " + data.shape_str() + ")";
  }
};
} // namespace tpipeline