/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {
namespace loss {

// CrossEntropy Loss (from probabilities)
template <typename T>
void compute_crossentropy_loss_probs(const T *predictions, const int *labels, float &loss,
                                     const size_t batch_size, const size_t num_classes, T epsilon,
                                     cudaStream_t stream);

template <typename T>
void compute_crossentropy_gradient_probs(const T *predictions, const int *labels, T *grad_output,
                                         const size_t batch_size, const size_t num_classes,
                                         T epsilon, cudaStream_t stream);

// CrossEntropy Loss (from logits)
template <typename T>
void compute_crossentropy_loss_logits(const T *logits, const int *labels, float &loss,
                                      const size_t batch_size, const size_t num_classes,
                                      cudaStream_t stream = 0);

template <typename T>
void compute_crossentropy_gradient_logits(const T *logits, const int *labels, T *grad_output,
                                          const size_t batch_size, const size_t num_classes,
                                          cudaStream_t stream = 0);

// MSE Loss
template <typename T>
void compute_mse_loss(const T *predictions, const T *targets, float &loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream);

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *grad_output,
                          const size_t batch_size, const size_t output_size, cudaStream_t stream);

// MAE Loss
template <typename T>
void compute_mae_loss(const T *predictions, const T *targets, float &loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream);

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *grad_output,
                          const size_t batch_size, const size_t output_size, cudaStream_t stream);

// Huber Loss
template <typename T>
void compute_huber_loss(const T *predictions, const T *targets, float &loss,
                        const size_t batch_size, const size_t output_size, T delta,
                        cudaStream_t stream);

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *grad_output,
                            const size_t batch_size, const size_t output_size, T delta,
                            cudaStream_t stream);

}  // namespace loss
}  // namespace cuda
}  // namespace tnn

#endif