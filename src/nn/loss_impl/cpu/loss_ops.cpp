/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/loss_impl/cpu/loss_ops.hpp"

#include "ops/ops.hpp"
#include "threading/thread_handler.hpp"
#include <algorithm>
#include <cmath>

namespace tnn {
namespace cpu {
namespace loss {

// CrossEntropy Loss Implementation
template <typename T>
T compute_crossentropy_loss(const T *predictions, const T *targets, const size_t batch_size,
                            const size_t num_classes, T epsilon) {
  double total_loss = 0.0;

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets[i * num_classes + j] > static_cast<T>(0.5)) {
        const T pred =
            std::clamp(predictions[i * num_classes + j], epsilon, static_cast<T>(1.0) - epsilon);
        total_loss -= std::log(pred);
        break;
      }
    }
  }

  return static_cast<T>(total_loss / batch_size);
}

template <typename T>
void compute_crossentropy_gradient(const T *predictions, const T *targets, T *gradient,
                                   const size_t batch_size, const size_t num_classes) {
  const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  parallel_for<size_t>(0, batch_size * num_classes, [&](size_t idx) {
    gradient[idx] = (predictions[idx] - targets[idx]) * inv_batch_size;
  });
}

// Softmax CrossEntropy Loss Implementation
template <typename T>
T compute_softmax_crossentropy_loss(const T *logits, const T *targets, const size_t batch_size,
                                    const size_t num_classes) {
  double total_loss = 0.0;

  for (size_t i = 0; i < batch_size; ++i) {
    T max_logit = logits[i * num_classes];
    for (size_t j = 1; j < num_classes; ++j) {
      max_logit = std::max(max_logit, logits[i * num_classes + j]);
    }

    double sum_exp = 0.0;
    for (size_t j = 0; j < num_classes; ++j) {
      sum_exp += std::exp(static_cast<double>(logits[i * num_classes + j] - max_logit));
    }
    const T log_sum_exp = static_cast<T>(std::log(sum_exp)) + max_logit;

    for (size_t j = 0; j < num_classes; ++j) {
      if (targets[i * num_classes + j] > static_cast<T>(0.5)) {
        total_loss += static_cast<double>(log_sum_exp - logits[i * num_classes + j]);
        break;
      }
    }
  }

  return static_cast<T>(total_loss / batch_size);
}

template <typename T>
void compute_softmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                           const size_t batch_size, const size_t num_classes) {
  const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    T max_logit = logits[i * num_classes];
    for (size_t j = 1; j < num_classes; ++j) {
      max_logit = std::max(max_logit, logits[i * num_classes + j]);
    }

    double sum_exp = 0.0;
    for (size_t j = 0; j < num_classes; ++j) {
      sum_exp += std::exp(static_cast<double>(logits[i * num_classes + j] - max_logit));
    }

    parallel_for<size_t>(0, num_classes, [&](size_t j) {
      const T softmax_prob = static_cast<T>(
          std::exp(static_cast<double>(logits[i * num_classes + j] - max_logit)) / sum_exp);
      gradient[i * num_classes + j] =
          (softmax_prob - targets[i * num_classes + j]) * inv_batch_size;
    });
  }
}

// MSE Loss Implementation
template <typename T>
T compute_mse_loss(const T *predictions, const T *targets, const size_t batch_size,
                   const size_t output_size) {
  double total_loss = 0.0;
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    const T diff = predictions[idx] - targets[idx];
    total_loss += static_cast<double>(diff * diff);
  }

  return static_cast<T>(total_loss / total_size);
}

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size) {
  const T scale = static_cast<T>(2.0) / static_cast<T>(batch_size * output_size);
  const size_t total_size = batch_size * output_size;

  parallel_for<size_t>(0, total_size, [&](size_t idx) {
    gradient[idx] = (predictions[idx] - targets[idx]) * scale;
  });
}

// MAE Loss Implementation
template <typename T>
T compute_mae_loss(const T *predictions, const T *targets, const size_t batch_size,
                   const size_t output_size) {
  double total_loss = 0.0;
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    total_loss += std::abs(predictions[idx] - targets[idx]);
  }

  return static_cast<T>(total_loss / total_size);
}

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size) {
  const T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
  const size_t total_size = batch_size * output_size;

  parallel_for<size_t>(0, total_size, [&](size_t idx) {
    const T diff = predictions[idx] - targets[idx];
    gradient[idx] = (diff > static_cast<T>(0) ? scale : -scale);
  });
}

// Huber Loss Implementation
template <typename T>
T compute_huber_loss(const T *predictions, const T *targets, const size_t batch_size,
                     const size_t output_size, T delta) {
  double total_loss = 0.0;
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    const T diff = std::abs(predictions[idx] - targets[idx]);
    if (diff <= delta) {
      total_loss += static_cast<double>(0.5 * diff * diff);
    } else {
      total_loss += static_cast<double>(delta * diff - 0.5 * delta * delta);
    }
  }

  return static_cast<T>(total_loss / total_size);
}

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *gradient,
                            const size_t batch_size, const size_t output_size, T delta) {
  const T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
  const size_t total_size = batch_size * output_size;

  parallel_for<size_t>(0, total_size, [&](size_t idx) {
    const T diff = predictions[idx] - targets[idx];
    const T abs_diff = std::abs(diff);

    if (abs_diff <= delta) {
      gradient[idx] = diff * scale;
    } else {
      gradient[idx] = (diff > static_cast<T>(0) ? delta : -delta) * scale;
    }
  });
}

// Explicit template instantiations
template float compute_crossentropy_loss<float>(const float *predictions, const float *targets,
                                                const size_t batch_size, const size_t num_classes,
                                                float epsilon);
template double compute_crossentropy_loss<double>(const double *predictions, const double *targets,
                                                  const size_t batch_size, const size_t num_classes,
                                                  double epsilon);
template void compute_crossentropy_gradient<float>(const float *predictions, const float *targets,
                                                   float *gradient, const size_t batch_size,
                                                   const size_t num_classes);
template void compute_crossentropy_gradient<double>(const double *predictions,
                                                    const double *targets, double *gradient,
                                                    const size_t batch_size,
                                                    const size_t num_classes);

template float compute_softmax_crossentropy_loss<float>(const float *logits, const float *targets,
                                                        const size_t batch_size,
                                                        const size_t num_classes);
template double compute_softmax_crossentropy_loss<double>(const double *logits,
                                                          const double *targets,
                                                          const size_t batch_size,
                                                          const size_t num_classes);
template void compute_softmax_crossentropy_gradient<float>(const float *logits,
                                                           const float *targets, float *gradient,
                                                           const size_t batch_size,
                                                           const size_t num_classes);
template void compute_softmax_crossentropy_gradient<double>(const double *logits,
                                                            const double *targets, double *gradient,
                                                            const size_t batch_size,
                                                            const size_t num_classes);

template float compute_mse_loss<float>(const float *predictions, const float *targets,
                                       const size_t batch_size, const size_t output_size);
template double compute_mse_loss<double>(const double *predictions, const double *targets,
                                         const size_t batch_size, const size_t output_size);
template void compute_mse_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size);
template void compute_mse_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size);

template float compute_mae_loss<float>(const float *predictions, const float *targets,
                                       const size_t batch_size, const size_t output_size);
template double compute_mae_loss<double>(const double *predictions, const double *targets,
                                         const size_t batch_size, const size_t output_size);
template void compute_mae_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size);
template void compute_mae_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size);

template float compute_huber_loss<float>(const float *predictions, const float *targets,
                                         const size_t batch_size, const size_t output_size,
                                         float delta);
template double compute_huber_loss<double>(const double *predictions, const double *targets,
                                           const size_t batch_size, const size_t output_size,
                                           double delta);
template void compute_huber_gradient<float>(const float *predictions, const float *targets,
                                            float *gradient, const size_t batch_size,
                                            const size_t output_size, float delta);
template void compute_huber_gradient<double>(const double *predictions, const double *targets,
                                             double *gradient, const size_t batch_size,
                                             const size_t output_size, double delta);

} // namespace loss
} // namespace cpu
} // namespace tnn
