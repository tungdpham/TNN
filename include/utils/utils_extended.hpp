#pragma once

#include "ops.hpp"
#include "tensor/tensor.hpp"

namespace utils {
template <typename T>
float compute_class_accuracy(const Tensor<float> &predictions, const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  int total_correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {

    int pred_class = 0;
    float max_pred = predictions(i, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      const float pred_val = predictions(i, j, 0, 0);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets(i, j, 0, 0) > 0.5f) {
        true_class = static_cast<int>(j);
        break;
      }
    }

    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return static_cast<float>(total_correct) / static_cast<float>(batch_size);
}

} // namespace utils