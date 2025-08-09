#pragma once

namespace tpipeline {
enum TaskType { Forward, Backward };

template <typename T = float> struct Task {
  TaskType type;
  Tensor<T> data;
  int micro_batch_id;

  Task(TaskType t, const Tensor<T> &d, int mb_id)
      : type(t), data(d), micro_batch_id(mb_id) {}
};
} // namespace tpipeline