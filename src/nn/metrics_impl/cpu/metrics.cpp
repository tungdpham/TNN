/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/metrics_impl/cpu/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace metrics {

template <typename T>
float compute_precision(const T* predictions, const int* targets, const size_t batch_size,
                        const size_t num_classes, int class_id) {
  if (class_id == -1) {
    // Macro-average precision across all classes
    float total_precision = 0.0f;
    int valid_classes = 0;

    for (int c = 0; c < static_cast<int>(num_classes); ++c) {
      float class_precision = compute_precision(predictions, targets, batch_size, num_classes, c);
      if (!std::isnan(class_precision)) {
        total_precision += class_precision;
        valid_classes++;
      }
    }

    return valid_classes > 0 ? total_precision / valid_classes : 0.0f;
  }

  // Compute precision for a specific class
  int true_positives = 0;
  int false_positives = 0;

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

    int true_class = targets[i];

    if (pred_class == class_id) {
      if (true_class == class_id) {
        true_positives++;
      } else {
        false_positives++;
      }
    }
  }

  int total_predicted_positive = true_positives + false_positives;
  return total_predicted_positive > 0
             ? static_cast<float>(true_positives) / total_predicted_positive
             : 0.0f;
}

template <typename T>
float compute_recall(const T* predictions, const int* targets, const size_t batch_size,
                     const size_t num_classes, int class_id) {
  if (class_id == -1) {
    // Macro-average recall across all classes
    float total_recall = 0.0f;
    int valid_classes = 0;

    for (int c = 0; c < static_cast<int>(num_classes); ++c) {
      float class_recall = compute_recall(predictions, targets, batch_size, num_classes, c);
      if (!std::isnan(class_recall)) {
        total_recall += class_recall;
        valid_classes++;
      }
    }

    return valid_classes > 0 ? total_recall / valid_classes : 0.0f;
  }

  // Compute recall for a specific class
  int true_positives = 0;
  int false_negatives = 0;

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

    int true_class = targets[i];

    if (true_class == class_id) {
      if (pred_class == class_id) {
        true_positives++;
      } else {
        false_negatives++;
      }
    }
  }

  int total_actual_positive = true_positives + false_negatives;
  return total_actual_positive > 0 ? static_cast<float>(true_positives) / total_actual_positive
                                   : 0.0f;
}

template <typename T>
float compute_f1_score(const T* predictions, const int* targets, const size_t batch_size,
                       const size_t num_classes, int class_id) {
  float precision = compute_precision(predictions, targets, batch_size, num_classes, class_id);
  float recall = compute_recall(predictions, targets, batch_size, num_classes, class_id);

  if (precision + recall > 0.0f) {
    return 2.0f * (precision * recall) / (precision + recall);
  }
  return 0.0f;
}

template <typename T>
float compute_perplexity(const T* predictions, const int* targets, const size_t batch_size,
                         const size_t num_classes) {
  double total_log_likelihood = 0.0;
  const double epsilon = 1e-10;  // To avoid log(0)

  for (size_t i = 0; i < batch_size; ++i) {
    int true_class = targets[i];
    double prob = static_cast<double>(predictions[i * num_classes + true_class]);
    prob = std::max(prob, epsilon);  // Avoid log(0)
    total_log_likelihood += std::log(prob);
  }

  double avg_log_likelihood = total_log_likelihood / static_cast<double>(batch_size);
  return static_cast<float>(std::exp(-avg_log_likelihood));
}

template <typename T>
float compute_top_k_accuracy(const T* predictions, const int* targets, const size_t batch_size,
                             const size_t num_classes, int k) {
  int correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    // Get the true class
    int true_class = targets[i];

    // Create pairs of (probability, class_index)
    std::vector<std::pair<double, int>> class_probs;
    class_probs.reserve(num_classes);

    for (size_t j = 0; j < num_classes; ++j) {
      class_probs.emplace_back(static_cast<double>(predictions[i * num_classes + j]),
                               static_cast<int>(j));
    }

    // Sort in descending order by probability
    std::partial_sort(
        class_probs.begin(), class_probs.begin() + std::min(k, static_cast<int>(num_classes)),
        class_probs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    // Check if true class is in top-k
    for (int j = 0; j < std::min(k, static_cast<int>(num_classes)); ++j) {
      if (class_probs[j].second == true_class) {
        correct++;
        break;
      }
    }
  }

  return static_cast<float>(correct) / static_cast<float>(batch_size);
}

template <typename T>
float compute_mae(const T* predictions, const T* targets, const size_t total_elements) {
  double total_error = 0.0;

  for (size_t i = 0; i < total_elements; ++i) {
    double diff = static_cast<double>(predictions[i]) - static_cast<double>(targets[i]);
    total_error += std::abs(diff);
  }

  return static_cast<float>(total_error / static_cast<double>(total_elements));
}

template <typename T>
float compute_mse(const T* predictions, const T* targets, const size_t total_elements) {
  double total_squared_error = 0.0;

  for (size_t i = 0; i < total_elements; ++i) {
    double diff = static_cast<double>(predictions[i]) - static_cast<double>(targets[i]);
    total_squared_error += diff * diff;
  }

  return static_cast<float>(total_squared_error / static_cast<double>(total_elements));
}

template <typename T>
int compute_class_corrects(const T* predictions, const int* targets, const size_t batch_size,
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

    int true_class = targets[i];

    if (pred_class == true_class) {
      total_correct++;
    }
  }

  return total_correct;
}

#define INSTANTIATE(T)                                                                        \
  template float compute_precision<T>(const T* predictions, const int* targets,               \
                                      const size_t batch_size, const size_t num_classes,      \
                                      int class_id);                                          \
  template float compute_recall<T>(const T* predictions, const int* targets,                  \
                                   const size_t batch_size, const size_t num_classes,         \
                                   int class_id);                                             \
  template float compute_f1_score<T>(const T* predictions, const int* targets,                \
                                     const size_t batch_size, const size_t num_classes,       \
                                     int class_id);                                           \
  template float compute_perplexity<T>(const T* predictions, const int* targets,              \
                                       const size_t batch_size, const size_t num_classes);    \
  template float compute_top_k_accuracy<T>(const T* predictions, const int* targets,          \
                                           const size_t batch_size, const size_t num_classes, \
                                           int k);                                            \
  template float compute_mae<T>(const T* predictions, const T* targets,                       \
                                const size_t total_elements);                                 \
  template float compute_mse<T>(const T* predictions, const T* targets,                       \
                                const size_t total_elements);                                 \
  template int compute_class_corrects<T>(const T* predictions, const int* targets,            \
                                         const size_t batch_size, const size_t num_classes,   \
                                         float threshold);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace metrics
}  // namespace cpu
}  // namespace tnn
