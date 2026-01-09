/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/loss_impl/cpu/loss_ops.hpp"

#include "threading/thread_handler.hpp"
#include <algorithm>
#include <cmath>

namespace tnn {
namespace cpu {
namespace loss {

template <typename T>
void compute_crossentropy_loss(const T *predictions, const T *targets, T &loss,
                               const size_t batch_size, const size_t num_classes, T epsilon,
                               const size_t spatial_dim) {
  double total_loss = 0.0;
  const size_t batch_stride = num_classes * spatial_dim;
  const size_t channel_stride = spatial_dim;

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t s = 0; s < spatial_dim; ++s) {
      size_t batch_start = i * batch_stride;
      for (size_t c = 0; c < num_classes; ++c) {
        size_t index = batch_start + c * channel_stride + s;
        if (targets[index] > static_cast<T>(0.5)) {
          const T pred = std::clamp(predictions[index], epsilon, static_cast<T>(1.0) - epsilon);
          total_loss -= std::log(pred);
          break;
        }
      }
    }
  }

  loss = static_cast<T>(total_loss / (batch_size * spatial_dim));
}

template <typename T>
void compute_crossentropy_gradient(const T *predictions, const T *targets, T *gradient,
                                   const size_t batch_size, const size_t num_classes,
                                   const size_t spatial_dim) {
  const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size * spatial_dim);

  parallel_for<size_t>(0, batch_size * num_classes * spatial_dim, [&](size_t idx) {
    gradient[idx] = (predictions[idx] - targets[idx]) * inv_batch_size;
  });
}

template <typename T>
void compute_softmax_crossentropy_loss(const T *logits, const T *targets, T &loss,
                                       const size_t batch_size, const size_t num_classes,
                                       const size_t spatial_dim) {
  double total_loss = 0.0;
  const size_t channel_stride = spatial_dim;
  const size_t batch_stride = num_classes * spatial_dim;

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t s = 0; s < spatial_dim; ++s) {
      size_t batch_start = i * batch_stride;

      T max_logit = logits[batch_start + 0 * channel_stride + s];
      for (size_t c = 1; c < num_classes; ++c) {
        max_logit = std::max(max_logit, logits[batch_start + c * channel_stride + s]);
      }

      double sum_exp = 0.0;
      for (size_t c = 0; c < num_classes; ++c) {
        sum_exp +=
            std::exp(static_cast<double>(logits[batch_start + c * channel_stride + s] - max_logit));
      }
      const T log_sum_exp = static_cast<T>(std::log(sum_exp)) + max_logit;

      for (size_t c = 0; c < num_classes; ++c) {
        size_t index = batch_start + c * channel_stride + s;
        if (targets[index] > static_cast<T>(0.5)) {
          total_loss += static_cast<double>(log_sum_exp - logits[index]);
          break;
        }
      }
    }
  }

  loss = static_cast<T>(total_loss / (batch_size * spatial_dim));
}

template <typename T>
void compute_softmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                           const size_t batch_size, const size_t num_classes,
                                           const size_t spatial_dim) {
  const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size * spatial_dim);
  const size_t channel_stride = spatial_dim;
  const size_t batch_stride = num_classes * spatial_dim;

  parallel_for<size_t>(0, batch_size * spatial_dim, [&](size_t idx) {
    size_t i = idx / spatial_dim;
    size_t s = idx % spatial_dim;
    size_t batch_start = i * batch_stride;

    T max_logit = logits[batch_start + 0 * channel_stride + s];
    for (size_t c = 1; c < num_classes; ++c) {
      max_logit = std::max(max_logit, logits[batch_start + c * channel_stride + s]);
    }

    double sum_exp = 0.0;
    for (size_t c = 0; c < num_classes; ++c) {
      sum_exp +=
          std::exp(static_cast<double>(logits[batch_start + c * channel_stride + s] - max_logit));
    }

    for (size_t c = 0; c < num_classes; ++c) {
      size_t current_idx = batch_start + c * channel_stride + s;
      const T softmax_prob =
          static_cast<T>(std::exp(static_cast<double>(logits[current_idx] - max_logit)) / sum_exp);
      gradient[current_idx] = (softmax_prob - targets[current_idx]) * inv_batch_size;
    }
  });
}

template <typename T>
void compute_mse_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                      const size_t output_size) {
  double total_loss = 0.0;
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    const T diff = predictions[idx] - targets[idx];
    total_loss += static_cast<double>(diff * diff);
  }

  loss = static_cast<T>(total_loss / total_size);
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

template <typename T>
void compute_logsoftmax_crossentropy_loss(const T *logits, const T *targets, T &loss,
                                          const size_t batch_size, const size_t num_classes,
                                          const size_t spatial_dim) {
  double total_loss = 0.0;
  const size_t channel_stride = spatial_dim;
  const size_t batch_stride = num_classes * spatial_dim;

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t s = 0; s < spatial_dim; ++s) {
      size_t batch_start = i * batch_stride;

      // Calculate max logit for numerical stability across classes at spatial position s
      T max_logit = logits[batch_start + 0 * channel_stride + s];
      for (size_t c = 1; c < num_classes; ++c) {
        max_logit = std::max(max_logit, logits[batch_start + c * channel_stride + s]);
      }

      double sum_exp = 0.0;
      for (size_t c = 0; c < num_classes; ++c) {
        sum_exp +=
            std::exp(static_cast<double>(logits[batch_start + c * channel_stride + s] - max_logit));
      }
      const T log_sum_exp = static_cast<T>(std::log(sum_exp)) + max_logit;

      // Find target class
      for (size_t c = 0; c < num_classes; ++c) {
        if (targets[batch_start + c * channel_stride + s] > static_cast<T>(0.5)) {
          total_loss +=
              static_cast<double>(log_sum_exp - logits[batch_start + c * channel_stride + s]);
          break;
        }
      }
    }
  }

  loss = static_cast<T>(total_loss / (batch_size * spatial_dim));
}

template <typename T>
void compute_logsoftmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                              const size_t batch_size, const size_t num_classes,
                                              const size_t spatial_dim) {
  const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size * spatial_dim);
  const size_t channel_stride = spatial_dim;
  const size_t batch_stride = num_classes * spatial_dim;

  // Parallelize over batch and spatial dimensions combined
  parallel_for<size_t>(0, batch_size * spatial_dim, [&](size_t idx) {
    size_t i = idx / spatial_dim;
    size_t s = idx % spatial_dim;
    size_t batch_start = i * batch_stride;

    // Numerical stability
    T max_logit = logits[batch_start + 0 * channel_stride + s];
    for (size_t c = 1; c < num_classes; ++c) {
      max_logit = std::max(max_logit, logits[batch_start + c * channel_stride + s]);
    }

    double sum_exp = 0.0;
    for (size_t c = 0; c < num_classes; ++c) {
      sum_exp +=
          std::exp(static_cast<double>(logits[batch_start + c * channel_stride + s] - max_logit));
    }

    for (size_t c = 0; c < num_classes; ++c) {
      size_t current_idx = batch_start + c * channel_stride + s;
      const T softmax_prob =
          static_cast<T>(std::exp(static_cast<double>(logits[current_idx] - max_logit)) / sum_exp);
      gradient[current_idx] = (softmax_prob - targets[current_idx]) * inv_batch_size;
    }
  });
}

template <typename T>
void compute_mae_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                      const size_t output_size) {
  double total_loss = 0.0;
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    total_loss += std::abs(predictions[idx] - targets[idx]);
  }

  loss = static_cast<T>(total_loss / total_size);
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

template <typename T>
void compute_huber_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
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

  loss = static_cast<T>(total_loss / total_size);
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

template void compute_crossentropy_loss<float>(const float *predictions, const float *targets,
                                               float &loss, const size_t batch_size,
                                               const size_t num_classes, float epsilon,
                                               const size_t spatial_dim);
template void compute_crossentropy_loss<double>(const double *predictions, const double *targets,
                                                double &loss, const size_t batch_size,
                                                const size_t num_classes, double epsilon,
                                                const size_t spatial_dim);
template void compute_crossentropy_gradient<float>(const float *predictions, const float *targets,
                                                   float *gradient, const size_t batch_size,
                                                   const size_t num_classes,
                                                   const size_t spatial_dim);
template void compute_crossentropy_gradient<double>(const double *predictions,
                                                    const double *targets, double *gradient,
                                                    const size_t batch_size,
                                                    const size_t num_classes,
                                                    const size_t spatial_dim);

template void compute_softmax_crossentropy_loss<float>(const float *logits, const float *targets,
                                                       float &loss, const size_t batch_size,
                                                       const size_t num_classes,
                                                       const size_t spatial_dim);
template void compute_softmax_crossentropy_loss<double>(const double *logits, const double *targets,
                                                        double &loss, const size_t batch_size,
                                                        const size_t num_classes,
                                                        const size_t spatial_dim);
template void compute_softmax_crossentropy_gradient<float>(const float *logits,
                                                           const float *targets, float *gradient,
                                                           const size_t batch_size,
                                                           const size_t num_classes,
                                                           const size_t spatial_dim);
template void compute_softmax_crossentropy_gradient<double>(const double *logits,
                                                            const double *targets, double *gradient,
                                                            const size_t batch_size,
                                                            const size_t num_classes,
                                                            const size_t spatial_dim);

template void compute_logsoftmax_crossentropy_loss<float>(const float *logits, const float *targets,
                                                          float &loss, const size_t batch_size,
                                                          const size_t num_classes,
                                                          const size_t spatial_dim);
template void compute_logsoftmax_crossentropy_loss<double>(const double *logits,
                                                           const double *targets, double &loss,
                                                           const size_t batch_size,
                                                           const size_t num_classes,
                                                           const size_t spatial_dim);
template void compute_logsoftmax_crossentropy_gradient<float>(const float *logits,
                                                              const float *targets, float *gradient,
                                                              const size_t batch_size,
                                                              const size_t num_classes,
                                                              const size_t spatial_dim);
template void compute_logsoftmax_crossentropy_gradient<double>(
    const double *logits, const double *targets, double *gradient, const size_t batch_size,
    const size_t num_classes, const size_t spatial_dim);

template void compute_mse_loss<float>(const float *predictions, const float *targets, float &loss,
                                      const size_t batch_size, const size_t output_size);
template void compute_mse_loss<double>(const double *predictions, const double *targets,
                                       double &loss, const size_t batch_size,
                                       const size_t output_size);
template void compute_mse_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size);
template void compute_mse_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size);

template void compute_mae_loss<float>(const float *predictions, const float *targets, float &loss,
                                      const size_t batch_size, const size_t output_size);
template void compute_mae_loss<double>(const double *predictions, const double *targets,
                                       double &loss, const size_t batch_size,
                                       const size_t output_size);
template void compute_mae_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size);
template void compute_mae_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size);

template void compute_huber_loss<float>(const float *predictions, const float *targets, float &loss,
                                        const size_t batch_size, const size_t output_size,
                                        float delta);
template void compute_huber_loss<double>(const double *predictions, const double *targets,
                                         double &loss, const size_t batch_size,
                                         const size_t output_size, double delta);
template void compute_huber_gradient<float>(const float *predictions, const float *targets,
                                            float *gradient, const size_t batch_size,
                                            const size_t output_size, float delta);
template void compute_huber_gradient<double>(const double *predictions, const double *targets,
                                             double *gradient, const size_t batch_size,
                                             const size_t output_size, double delta);

} // namespace loss
} // namespace cpu
} // namespace tnn
