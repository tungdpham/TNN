/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/loss_impl/cpu/loss_ops.hpp"

#include <algorithm>
#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace loss {

template <typename T>
void compute_crossentropy_loss_probs(const T *predictions, const int *labels, float &loss,
                                     const size_t batch_size, const size_t num_classes, T epsilon) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  ComputeT total_loss = static_cast<ComputeT>(0);
  const size_t batch_stride = num_classes;

  for (size_t i = 0; i < batch_size; ++i) {
    size_t batch_start = i * batch_stride;
    int label = labels[i];
    size_t index = batch_start + label;

    const ComputeT pred =
        std::clamp(static_cast<ComputeT>(predictions[index]), static_cast<ComputeT>(epsilon),
                   static_cast<ComputeT>(1.0) - static_cast<ComputeT>(epsilon));
    total_loss -= std::log(pred);
  }

  loss = static_cast<float>(total_loss / static_cast<ComputeT>(batch_size));
}

template <typename T>
void compute_crossentropy_gradient_probs(const T *predictions, const int *labels, T *grad_output,
                                         const size_t batch_size, const size_t num_classes,
                                         T epsilon) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  const ComputeT inv_batch_size = static_cast<ComputeT>(1.0) / static_cast<ComputeT>(batch_size);

  parallel_for<size_t>(0, batch_size * num_classes, [&](size_t idx) {
    size_t b = idx / num_classes;
    size_t c = idx % num_classes;
    int label = labels[b];

    if (c == static_cast<size_t>(label)) {
      ComputeT pred = static_cast<ComputeT>(predictions[idx]);
      ComputeT eps = static_cast<ComputeT>(epsilon);
      pred = std::clamp(pred, eps, ComputeT(1.0) - eps);
      grad_output[idx] = static_cast<T>(-ComputeT(1.0) / pred * inv_batch_size);
    } else {
      grad_output[idx] = static_cast<T>(0.0);
    }
  });
}

template <typename T>
void compute_mse_loss(const T *predictions, const T *targets, float &loss, const size_t batch_size,
                      const size_t output_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  ComputeT total_loss = static_cast<ComputeT>(0);
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    const ComputeT diff =
        static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
    total_loss += diff * diff;
  }

  loss = static_cast<float>(total_loss / static_cast<ComputeT>(total_size));
}

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *grad_output,
                          const size_t batch_size, const size_t output_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  const ComputeT scale =
      static_cast<ComputeT>(2.0) / static_cast<ComputeT>(batch_size * output_size);
  const size_t total_size = batch_size * output_size;

  parallel_for<size_t>(0, total_size, [&](size_t idx) {
    grad_output[idx] = static_cast<T>(
        (static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx])) * scale);
  });
}

template <typename T>
void compute_crossentropy_loss_logits(const T *logits, const int *labels, float &loss,
                                      const size_t batch_size, const size_t num_classes) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  ComputeT total_loss = static_cast<ComputeT>(0);
  const size_t batch_stride = num_classes;

  for (size_t i = 0; i < batch_size; ++i) {
    size_t batch_start = i * batch_stride;
    int label = labels[i];

    ComputeT max_logit = static_cast<ComputeT>(logits[batch_start + 0]);
    for (size_t c = 1; c < num_classes; ++c) {
      max_logit = std::max(max_logit, static_cast<ComputeT>(logits[batch_start + c]));
    }

    ComputeT sum_exp = static_cast<ComputeT>(0);
    for (size_t c = 0; c < num_classes; ++c) {
      sum_exp += std::exp(static_cast<ComputeT>(logits[batch_start + c]) - max_logit);
    }
    const ComputeT log_sum_exp = std::log(sum_exp) + max_logit;

    total_loss += log_sum_exp - static_cast<ComputeT>(logits[batch_start + label]);
  }

  loss = static_cast<float>(total_loss / static_cast<ComputeT>(batch_size));
}

template <typename T>
void compute_crossentropy_gradient_logits(const T *logits, const int *labels, T *grad_output,
                                          const size_t batch_size, const size_t num_classes) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  const ComputeT inv_batch_size = static_cast<ComputeT>(1.0) / static_cast<ComputeT>(batch_size);
  const size_t batch_stride = num_classes;

  parallel_for<size_t>(0, batch_size, [&](size_t i) {
    size_t batch_start = i * batch_stride;
    int label = labels[i];

    ComputeT max_logit = static_cast<ComputeT>(logits[batch_start + 0]);
    for (size_t c = 1; c < num_classes; ++c) {
      max_logit = std::max(max_logit, static_cast<ComputeT>(logits[batch_start + c]));
    }

    ComputeT sum_exp = static_cast<ComputeT>(0);
    for (size_t c = 0; c < num_classes; ++c) {
      sum_exp += std::exp(static_cast<ComputeT>(logits[batch_start + c]) - max_logit);
    }

    for (size_t c = 0; c < num_classes; ++c) {
      size_t current_idx = batch_start + c;
      const ComputeT softmax_prob =
          std::exp(static_cast<ComputeT>(logits[current_idx]) - max_logit) / sum_exp;
      const ComputeT target = (c == static_cast<size_t>(label)) ? ComputeT(1.0) : ComputeT(0.0);
      grad_output[current_idx] = static_cast<T>((softmax_prob - target) * inv_batch_size);
    }
  });
}

template <typename T>
void compute_mae_loss(const T *predictions, const T *targets, float &loss, const size_t batch_size,
                      const size_t output_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  ComputeT total_loss = static_cast<ComputeT>(0);
  const size_t total_size = batch_size * output_size;

  for (size_t idx = 0; idx < total_size; ++idx) {
    total_loss +=
        std::fabs(static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]));
  }

  loss = static_cast<float>(total_loss / static_cast<ComputeT>(total_size));
}

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *grad_output,
                          const size_t batch_size, const size_t output_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  const ComputeT scale =
      static_cast<ComputeT>(1.0) / static_cast<ComputeT>(batch_size * output_size);
  const size_t total_size = batch_size * output_size;

  parallel_for<size_t>(0, total_size, [&](size_t idx) {
    const ComputeT diff =
        static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
    grad_output[idx] = static_cast<T>(diff > static_cast<ComputeT>(0) ? scale : -scale);
  });
}

template <typename T>
void compute_huber_loss(const T *predictions, const T *targets, float &loss,
                        const size_t batch_size, const size_t output_size, T delta) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  ComputeT total_loss = static_cast<ComputeT>(0);
  const size_t total_size = batch_size * output_size;
  const ComputeT delta_c = static_cast<ComputeT>(delta);

  for (size_t idx = 0; idx < total_size; ++idx) {
    const ComputeT diff =
        std::fabs(static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]));
    if (diff <= delta_c) {
      total_loss += static_cast<ComputeT>(0.5) * diff * diff;
    } else {
      total_loss += delta_c * diff - static_cast<ComputeT>(0.5) * delta_c * delta_c;
    }
  }

  loss = static_cast<float>(total_loss / static_cast<ComputeT>(total_size));
}

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *grad_output,
                            const size_t batch_size, const size_t output_size, T delta) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  const ComputeT scale =
      static_cast<ComputeT>(1.0) / static_cast<ComputeT>(batch_size * output_size);
  const size_t total_size = batch_size * output_size;
  const ComputeT delta_c = static_cast<ComputeT>(delta);

  parallel_for<size_t>(0, total_size, [&](size_t idx) {
    const ComputeT diff =
        static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
    const ComputeT abs_diff = std::fabs(diff);

    if (abs_diff <= delta_c) {
      grad_output[idx] = static_cast<T>(diff * scale);
    } else {
      grad_output[idx] =
          static_cast<T>((diff > static_cast<ComputeT>(0) ? delta_c : -delta_c) * scale);
    }
  });
}

#define INSTANTIATE(T)                                                                             \
  template void compute_crossentropy_loss_probs<T>(const T *predictions, const int *labels,        \
                                                   float &loss, const size_t batch_size,           \
                                                   const size_t num_classes, T epsilon);           \
                                                                                                   \
  template void compute_crossentropy_gradient_probs<T>(const T *predictions, const int *labels,    \
                                                       T *grad_output, const size_t batch_size,    \
                                                       const size_t num_classes, T epsilon);       \
                                                                                                   \
  template void compute_crossentropy_loss_logits<T>(const T *logits, const int *labels,            \
                                                    float &loss, const size_t batch_size,          \
                                                    const size_t num_classes);                     \
                                                                                                   \
  template void compute_crossentropy_gradient_logits<T>(const T *logits, const int *labels,        \
                                                        T *grad_output, const size_t batch_size,   \
                                                        const size_t num_classes);                 \
                                                                                                   \
  template void compute_mse_loss<T>(const T *predictions, const T *targets, float &loss,           \
                                    const size_t batch_size, const size_t output_size);            \
                                                                                                   \
  template void compute_mse_gradient<T>(const T *predictions, const T *targets, T *grad_output,    \
                                        const size_t batch_size, const size_t output_size);        \
                                                                                                   \
  template void compute_mae_loss<T>(const T *predictions, const T *targets, float &loss,           \
                                    const size_t batch_size, const size_t output_size);            \
                                                                                                   \
  template void compute_mae_gradient<T>(const T *predictions, const T *targets, T *grad_output,    \
                                        const size_t batch_size, const size_t output_size);        \
                                                                                                   \
  template void compute_huber_loss<T>(const T *predictions, const T *targets, float &loss,         \
                                      const size_t batch_size, const size_t output_size, T delta); \
                                                                                                   \
  template void compute_huber_gradient<T>(const T *predictions, const T *targets, T *grad_output,  \
                                          const size_t batch_size, const size_t output_size,       \
                                          T delta);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace loss
}  // namespace cpu
}  // namespace tnn
