/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cpu/adam_kernels.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
namespace adam {

template <typename T>
void update_adam(T *params_data, const T *grads_data, T *m_data, T *v_data, const size_t size,
                 const float learning_rate, const float beta1, const float beta2,
                 const float epsilon, const float bias_correction1, const float bias_correction2,
                 const float weight_decay, const bool decouple_weight_decay) {

  const T one_minus_beta1 = T(1.0) - static_cast<T>(beta1);
  const T one_minus_beta2 = T(1.0) - static_cast<T>(beta2);

  parallel_for<size_t>(0, size, [&](size_t i) {
    T grad = grads_data[i];

    m_data[i] = static_cast<T>(beta1) * m_data[i] + one_minus_beta1 * grad;

    v_data[i] = static_cast<T>(beta2) * v_data[i] + one_minus_beta2 * grad * grad;

    T m_hat = m_data[i] / static_cast<T>(bias_correction1);
    T v_hat = v_data[i] / static_cast<T>(bias_correction2);

    T update = static_cast<T>(learning_rate) * m_hat /
               (static_cast<T>(std::sqrt(static_cast<float>(v_hat))) + static_cast<T>(epsilon));

    if (weight_decay > 0.0f) {
      if (decouple_weight_decay) {

        params_data[i] -=
            static_cast<T>(weight_decay) * static_cast<T>(learning_rate) * params_data[i];
      } else {

        update += static_cast<T>(weight_decay) * static_cast<T>(learning_rate) * params_data[i];
      }
    }

    params_data[i] -= update;
  });
}

#define INSTANTIATE_ADAM(T)                                                                        \
  template void update_adam<T>(T * params_data, const T *grads_data, T *m_data, T *v_data,         \
                               const size_t size, const float learning_rate, const float beta1,    \
                               const float beta2, const float epsilon,                             \
                               const float bias_correction1, const float bias_correction2,         \
                               const float weight_decay, const bool decouple_weight_decay);
INSTANTIATE_ADAM(fp16)
INSTANTIATE_ADAM(float)
INSTANTIATE_ADAM(double)
#undef INSTANTIATE_ADAM

} // namespace adam
} // namespace cpu
} // namespace tnn
