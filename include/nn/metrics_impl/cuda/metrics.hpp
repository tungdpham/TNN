/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {
namespace metrics {

// Compute precision for classification
template <typename T>
float compute_precision(const T *predictions, const int *targets, const size_t batch_size,
                        const size_t num_classes, int class_id = -1, cudaStream_t stream = 0);

// Compute recall for classification
template <typename T>
float compute_recall(const T *predictions, const int *targets, const size_t batch_size,
                     const size_t num_classes, int class_id = -1, cudaStream_t stream = 0);

// Compute F1 score for classification
template <typename T>
float compute_f1_score(const T *predictions, const int *targets, const size_t batch_size,
                       const size_t num_classes, int class_id = -1, cudaStream_t stream = 0);

// Compute perplexity (common for language models)
template <typename T>
float compute_perplexity(const T *predictions, const int *targets, const size_t batch_size,
                         const size_t num_classes, cudaStream_t stream = 0);

// Compute top-K accuracy
template <typename T>
float compute_top_k_accuracy(const T *predictions, const int *targets, const size_t batch_size,
                             const size_t num_classes, int k = 5, cudaStream_t stream = 0);

// Compute Mean Absolute Error
template <typename T>
float compute_mae(const T *predictions, const T *targets, const size_t total_elements,
                  cudaStream_t stream = 0);

// Compute Mean Squared Error
template <typename T>
float compute_mse(const T *predictions, const T *targets, const size_t total_elements,
                  cudaStream_t stream = 0);

// Compute class corrects (number of correct predictions)
template <typename T>
int compute_class_corrects(const T *predictions, const int *targets, const size_t batch_size,
                           const size_t num_classes, float threshold = 0.5f,
                           cudaStream_t stream = 0);

}  // namespace metrics
}  // namespace cuda
}  // namespace tnn
