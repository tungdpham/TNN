/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "metrics_impl/cpu/metrics.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "metrics_impl/cuda/metrics.hpp"
#endif

namespace tnn {

namespace detail {

// Precision implementation
template <typename T>
inline float compute_precision_impl(const ConstTensor &predictions, const ConstTensor &targets,
                                    size_t batch_size, size_t num_classes, int class_id) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_precision(predictions->data_as<T>(), targets->data_as<int>(),
                                           batch_size, num_classes, class_id);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_precision(predictions->data_as<T>(), targets->data_as<int>(),
                                            batch_size, num_classes, class_id);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_precision.");
}

// Recall implementation
template <typename T>
inline float compute_recall_impl(const ConstTensor &predictions, const ConstTensor &targets,
                                 size_t batch_size, size_t num_classes, int class_id) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_recall(predictions->data_as<T>(), targets->data_as<int>(),
                                        batch_size, num_classes, class_id);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_recall(predictions->data_as<T>(), targets->data_as<int>(),
                                         batch_size, num_classes, class_id);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_recall.");
}

// F1 Score implementation
template <typename T>
inline float compute_f1_score_impl(const ConstTensor &predictions, const ConstTensor &targets,
                                   size_t batch_size, size_t num_classes, int class_id) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_f1_score(predictions->data_as<T>(), targets->data_as<int>(),
                                          batch_size, num_classes, class_id);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_f1_score(predictions->data_as<T>(), targets->data_as<int>(),
                                           batch_size, num_classes, class_id);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_f1_score.");
}

// Perplexity implementation
template <typename T>
inline float compute_perplexity_impl(const ConstTensor &predictions, const ConstTensor &targets,
                                     size_t batch_size, size_t num_classes) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_perplexity(predictions->data_as<T>(), targets->data_as<int>(),
                                            batch_size, num_classes);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_perplexity(predictions->data_as<T>(), targets->data_as<int>(),
                                             batch_size, num_classes);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_perplexity.");
}

// Top-K Accuracy implementation
template <typename T>
inline float compute_top_k_accuracy_impl(const ConstTensor &predictions, const ConstTensor &targets,
                                         size_t batch_size, size_t num_classes, int k) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_top_k_accuracy(predictions->data_as<T>(), targets->data_as<int>(),
                                                batch_size, num_classes, k);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_top_k_accuracy(predictions->data_as<T>(), targets->data_as<int>(),
                                                 batch_size, num_classes, k);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_top_k_accuracy.");
}

// Mean Absolute Error implementation
template <typename T>
inline float compute_mae_impl(const ConstTensor &predictions, const ConstTensor &targets,
                              size_t total_elements) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_mae(predictions->data_as<T>(), targets->data_as<T>(),
                                     total_elements);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_mae(predictions->data_as<T>(), targets->data_as<T>(),
                                      total_elements);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_mae.");
}

// Mean Squared Error implementation
template <typename T>
inline float compute_mse_impl(const ConstTensor &predictions, const ConstTensor &targets,
                              size_t total_elements) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_mse(predictions->data_as<T>(), targets->data_as<T>(),
                                     total_elements);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_mse(predictions->data_as<T>(), targets->data_as<T>(),
                                      total_elements);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_mse.");
}

// Class corrects implementation
template <typename T>
inline int compute_class_corrects_impl(const ConstTensor &predictions, const ConstTensor &targets,
                                       size_t batch_size, size_t num_classes, float threshold) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::metrics::compute_class_corrects(predictions->data_as<T>(), targets->data_as<int>(),
                                                batch_size, num_classes, threshold);
  }
#ifdef USE_CUDA
  else {
    return cuda::metrics::compute_class_corrects(predictions->data_as<T>(), targets->data_as<int>(),
                                                 batch_size, num_classes, threshold);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_class_corrects.");
}

}  // namespace detail

/**
 * Compute precision for a specific class
 * @param predictions Tensor of shape [batch_size, num_classes] containing predicted probabilities
 * @param targets Tensor of shape [batch_size] containing ground truth class indices
 * @param class_id The class ID to compute precision for (-1 for macro-average across all classes)
 * @return Precision score
 */
inline float compute_precision(const ConstTensor &predictions, const ConstTensor &targets,
                               int class_id = -1) {
  size_t batch_size = 1;
  for (size_t i = 0; i < predictions->dims() - 1; ++i) {
    batch_size *= predictions->shape()[i];
  }
  const size_t num_classes = predictions->shape().back();

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_precision_impl<T>(predictions, targets, batch_size,
                                                          num_classes, class_id));
}

/**
 * Compute recall for a specific class
 * @param predictions Tensor of shape [batch_size, num_classes] containing predicted probabilities
 * @param targets Tensor of shape [batch_size] containing ground truth class indices
 * @param class_id The class ID to compute recall for (-1 for macro-average across all classes)
 * @return Recall score
 */
inline float compute_recall(const ConstTensor &predictions, const ConstTensor &targets,
                            int class_id = -1) {
  size_t batch_size = 1;
  for (size_t i = 0; i < predictions->dims() - 1; ++i) {
    batch_size *= predictions->shape()[i];
  }
  const size_t num_classes = predictions->shape().back();

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_recall_impl<T>(predictions, targets, batch_size,
                                                       num_classes, class_id));
}

/**
 * Compute F1 score for a specific class
 * @param predictions Tensor of shape [batch_size, num_classes] containing predicted probabilities
 * @param targets Tensor of shape [batch_size] containing ground truth class indices
 * @param class_id The class ID to compute F1 score for (-1 for macro-average across all classes)
 * @return F1 score
 */
inline float compute_f1_score(const ConstTensor &predictions, const ConstTensor &targets,
                              int class_id = -1) {
  size_t batch_size = 1;
  for (size_t i = 0; i < predictions->dims() - 1; ++i) {
    batch_size *= predictions->shape()[i];
  }
  const size_t num_classes = predictions->shape().back();

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_f1_score_impl<T>(predictions, targets, batch_size,
                                                         num_classes, class_id));
}

/**
 * Compute perplexity - commonly used for language models
 * @param predictions Tensor of shape [batch_size, num_classes] containing predicted probabilities
 * @param targets Tensor of shape [batch_size] containing ground truth class indices
 * @return Perplexity score
 */
inline float compute_perplexity(const ConstTensor &predictions, const ConstTensor &targets) {
  size_t batch_size = 1;
  for (size_t i = 0; i < predictions->dims() - 1; ++i) {
    batch_size *= predictions->shape()[i];
  }
  const size_t num_classes = predictions->shape().back();

  DISPATCH_DTYPE(
      predictions->data_type(), T,
      return detail::compute_perplexity_impl<T>(predictions, targets, batch_size, num_classes));
}

/**
 * Compute top-K accuracy
 * @param predictions Tensor of shape [batch_size, num_classes] containing predicted probabilities
 * @param targets Tensor of shape [batch_size] containing ground truth class indices
 * @param k Number of top predictions to consider
 * @return Top-K accuracy
 */
inline float compute_top_k_accuracy(const ConstTensor &predictions, const ConstTensor &targets,
                                    int k = 5) {
  size_t batch_size = 1;
  for (size_t i = 0; i < predictions->dims() - 1; ++i) {
    batch_size *= predictions->shape()[i];
  }
  const size_t num_classes = predictions->shape().back();

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_top_k_accuracy_impl<T>(predictions, targets, batch_size,
                                                               num_classes, k));
}

/**
 * Compute Mean Absolute Error
 * @param predictions Tensor containing predicted values
 * @param targets Tensor containing ground truth values
 * @return MAE score
 */
inline float compute_mae(const ConstTensor &predictions, const ConstTensor &targets) {
  size_t total_elements = 1;
  for (size_t i = 0; i < predictions->dims(); ++i) {
    total_elements *= predictions->shape()[i];
  }

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_mae_impl<T>(predictions, targets, total_elements));
}

/**
 * Compute Mean Squared Error
 * @param predictions Tensor containing predicted values
 * @param targets Tensor containing ground truth values
 * @return MSE score
 */
inline float compute_mse(const ConstTensor &predictions, const ConstTensor &targets) {
  size_t total_elements = 1;
  for (size_t i = 0; i < predictions->dims(); ++i) {
    total_elements *= predictions->shape()[i];
  }

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_mse_impl<T>(predictions, targets, total_elements));
}

/**
 * Compute Root Mean Squared Error
 * @param predictions Tensor containing predicted values
 * @param targets Tensor containing ground truth values
 * @return RMSE score
 */
inline float compute_rmse(const ConstTensor &predictions, const ConstTensor &targets) {
  return std::sqrt(compute_mse(predictions, targets));
}

/**
 * Compute the number of correct class predictions
 * @param predictions Tensor of shape [batch_size, num_classes] containing predicted probabilities
 * @param targets Tensor of shape [batch_size] containing ground truth class indices
 * @param threshold Threshold for classification (default: 0.0, not used for argmax)
 * @return Number of correct predictions
 */
inline int compute_class_corrects(const ConstTensor &predictions, const ConstTensor &targets,
                                  float threshold = 0.0f) {
  size_t batch_size = 1;
  for (size_t i = 0; i < predictions->dims() - 1; ++i) {
    batch_size *= predictions->shape()[i];
  }
  const size_t num_classes = predictions->shape().back();

  DISPATCH_DTYPE(predictions->data_type(), T,
                 return detail::compute_class_corrects_impl<T>(predictions, targets, batch_size,
                                                               num_classes, threshold));
}

}  // namespace tnn
