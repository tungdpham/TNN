/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/accuracy_impl/cpu/accuracy.hpp"

#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace accuracy {

template <typename T>
float compute_class_accuracy(const T *predictions, const T *targets, const size_t batch_size,
                             const size_t num_classes) {
  int total_correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    int pred_class = 0;
    double max_pred = static_cast<double>(predictions[i * num_classes]);
    for (size_t j = 1; j < num_classes; ++j) {
      const double pred_val = static_cast<double>(predictions[i * num_classes + j]);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (static_cast<double>(targets[i * num_classes + j]) > 0.5) {
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

template <typename T>
int compute_class_corrects(const T *predictions, const T *targets, const size_t batch_size,
                           const size_t num_classes, float threshold) {
  int total_correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    int pred_class = 0;
    double max_pred = static_cast<double>(predictions[i * num_classes]);
    for (size_t j = 1; j < num_classes; ++j) {
      const double pred_val = static_cast<double>(predictions[i * num_classes + j]);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (static_cast<double>(targets[i * num_classes + j]) > static_cast<double>(threshold)) {
        true_class = static_cast<int>(j);
        break;
      }
    }

    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return total_correct;
}

template float compute_class_accuracy<float>(const float *, const float *, const size_t,
                                             const size_t);
template float compute_class_accuracy<double>(const double *, const double *, const size_t,
                                              const size_t);
template float compute_class_accuracy<fp16>(const fp16 *, const fp16 *, const size_t, const size_t);
template float compute_class_accuracy<bf16>(const bf16 *, const bf16 *, const size_t, const size_t);

template int compute_class_corrects<float>(const float *, const float *, const size_t, const size_t,
                                           float);
template int compute_class_corrects<double>(const double *, const double *, const size_t,
                                            const size_t, float);
template int compute_class_corrects<fp16>(const fp16 *, const fp16 *, const size_t, const size_t,
                                          float);
template int compute_class_corrects<bf16>(const bf16 *, const bf16 *, const size_t, const size_t,
                                          float);

}  // namespace accuracy
}  // namespace cpu
}  // namespace tnn
