#pragma once

#include <string>
#include "../tensor/tensor.hpp"

namespace tpipeline {
enum TaskType { FORWARD, BACKWARD };

template <typename T = float> struct Task {
  TaskType type;
  Tensor<T> data;
  size_t micro_batch_id;

  Task(TaskType t, const Tensor<T> &d, size_t mb_id)
      : type(t), data(d), micro_batch_id(mb_id) {}

  Task operator=(const Task &other) {
    if (this != &other) {
      type = other.type;
      data = other.data.clone();
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