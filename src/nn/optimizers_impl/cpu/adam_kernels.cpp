/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cpu/adam_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
namespace adam {

template <typename T>
void update_adam(T *params_data, const T *grads_data, T *m_data, T *v_data, const size_t size,
                 const float learning_rate, const float beta1, const float beta2,
                 const float epsilon, const float bias_correction1, const float bias_correction2,
                 const float weight_decay, const bool decouple_weight_decay) {

  const T one_minus_beta1 = static_cast<T>(1.0) - beta1;
  const T one_minus_beta2 = static_cast<T>(1.0) - beta2;

  parallel_for<size_t>(0, size, [&](size_t i) {
    T grad = grads_data[i];

    m_data[i] = beta1 * m_data[i] + one_minus_beta1 * grad;

    v_data[i] = beta2 * v_data[i] + one_minus_beta2 * grad * grad;

    T m_hat = m_data[i] / bias_correction1;

    T v_hat = v_data[i] / bias_correction2;

    T update = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

    if (weight_decay > 0.0f) {
      if (decouple_weight_decay) {

        params_data[i] -= weight_decay * learning_rate * params_data[i];
      } else {

        update += weight_decay * learning_rate * params_data[i];
      }
    }

    params_data[i] -= update;
  });
}

template void update_adam<float>(float *params_data, const float *grads_data, float *m_data,
                                 float *v_data, const size_t size, const float learning_rate,
                                 const float beta1, const float beta2, const float epsilon,
                                 const float bias_correction1, const float bias_correction2,
                                 const float weight_decay, const bool decouple_weight_decay);
template void update_adam<double>(double *params_data, const double *grads_data, double *m_data,
                                  double *v_data, const size_t size, const float learning_rate,
                                  const float beta1, const float beta2, const float epsilon,
                                  const float bias_correction1, const float bias_correction2,
                                  const float weight_decay, const bool decouple_weight_decay);

} // namespace adam
} // namespace cpu
} // namespace tnn
