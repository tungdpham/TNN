/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace loss {

// CrossEntropy Loss
template <typename T>
T compute_crossentropy_loss(const T *predictions, const T *targets, const size_t batch_size,
                            const size_t num_classes, T epsilon);

template <typename T>
void compute_crossentropy_gradient(const T *predictions, const T *targets, T *gradient,
                                   const size_t batch_size, const size_t num_classes);

// Softmax CrossEntropy Loss
template <typename T>
T compute_softmax_crossentropy_loss(const T *logits, const T *targets, const size_t batch_size,
                                    const size_t num_classes);

template <typename T>
void compute_softmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                           const size_t batch_size, const size_t num_classes);

// MSE Loss
template <typename T>
T compute_mse_loss(const T *predictions, const T *targets, const size_t batch_size,
                   const size_t output_size);

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size);

// MAE Loss
template <typename T>
T compute_mae_loss(const T *predictions, const T *targets, const size_t batch_size,
                   const size_t output_size);

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size);

// Huber Loss
template <typename T>
T compute_huber_loss(const T *predictions, const T *targets, const size_t batch_size,
                     const size_t output_size, T delta);

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *gradient,
                            const size_t batch_size, const size_t output_size, T delta);

} // namespace loss
} // namespace cpu
} // namespace tnn
