/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "utils/accuracy_impl/cpu/accuracy.hpp"

namespace tnn {
namespace cpu {
namespace accuracy {

float compute_class_accuracy(const float *predictions, const float *targets,
                             const size_t batch_size, const size_t num_classes) {
  int total_correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    int pred_class = 0;
    float max_pred = predictions[i * num_classes];
    for (size_t j = 1; j < num_classes; ++j) {
      const float pred_val = predictions[i * num_classes + j];
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets[i * num_classes + j] > 0.5f) {
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

float compute_class_corrects(const float *predictions, const float *targets,
                             const size_t batch_size, const size_t num_classes, float threshold) {
  int total_correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    int pred_class = 0;
    float max_pred = predictions[i * num_classes];
    for (size_t j = 1; j < num_classes; ++j) {
      const float pred_val = predictions[i * num_classes + j];
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets[i * num_classes + j] > threshold) {
        true_class = static_cast<int>(j);
        break;
      }
    }

    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return static_cast<float>(total_correct);
}

} // namespace accuracy
} // namespace cpu
} // namespace tnn
