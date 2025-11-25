/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace loss {

// CrossEntropy Loss
template <typename T>
void compute_crossentropy_loss(const T *predictions, const T *targets, T &loss,
                               const size_t batch_size, const size_t num_classes, T epsilon,
                               cudaStream_t stream);

template <typename T>
void compute_crossentropy_gradient(const T *predictions, const T *targets, T *gradient,
                                   const size_t batch_size, const size_t num_classes,
                                   cudaStream_t stream);

// Softmax CrossEntropy Loss
template <typename T>
void compute_softmax_crossentropy_loss(const T *logits, const T *targets, T &loss,
                                       const size_t batch_size, const size_t num_classes,
                                       cudaStream_t stream);

template <typename T>
void compute_softmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                           const size_t batch_size, const size_t num_classes,
                                           cudaStream_t stream);

// LogSoftmax CrossEntropy Loss
template <typename T>
void compute_logsoftmax_crossentropy_loss(const T *logits, const T *targets, T &loss,
                                          const size_t batch_size, const size_t num_classes,
                                          cudaStream_t stream);

template <typename T>
void compute_logsoftmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                              const size_t batch_size, const size_t num_classes,
                                              cudaStream_t stream);

// MSE Loss
template <typename T>
void compute_mse_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream);

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size, cudaStream_t stream);

// MAE Loss
template <typename T>
void compute_mae_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream);

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size, cudaStream_t stream);

// Huber Loss
template <typename T>
void compute_huber_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                        const size_t output_size, T delta, cudaStream_t stream);

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *gradient,
                            const size_t batch_size, const size_t output_size, T delta,
                            cudaStream_t stream);

} // namespace loss
} // namespace cuda
} // namespace tnn

#endif