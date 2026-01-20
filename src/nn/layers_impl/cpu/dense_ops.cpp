/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/dense_ops.hpp"

#include "math/gemm.hpp"
#include "ops/ops.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dense {
template <typename T>
void compute_dense_forward(const T *input_data, const T *weight_data, T *output_data,
                           const size_t batch_size, const size_t input_features,
                           const size_t output_features) {
  cpu::gemm<T>(input_data, weight_data, output_data, batch_size, output_features, input_features,
               false, true, T(1.0), T(0.0));
}

template <typename T>
void compute_weight_gradients(const T *input_data, const T *gradient_data, T *weight_grad_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features) {
  cpu::gemm<T>(gradient_data, input_data, weight_grad_data, output_features, input_features,
               batch_size, true, false, T(1.0), T(1.0));
}

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *grad_input_data,
                             const size_t batch_size, const size_t input_features,
                             const size_t output_features) {
  cpu::gemm<T>(gradient_data, weight_data, grad_input_data, batch_size, input_features,
               output_features, false, false, T(1.0), T(0.0));
}

template <typename T>
void compute_bias_gradients(const T *current_grad_data, T *bias_gradient_data,
                            const size_t batch_size, const size_t output_features) {
  parallel_for<size_t>(0, output_features, [&](size_t out_f) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      grad_sum += current_grad_data[n * output_features + out_f];
    }
    bias_gradient_data[out_f] += grad_sum;
  });
}

template <typename T>
void add_bias_vector(T *output_data, const T *bias_data, const size_t batch_size,
                     const size_t output_features) {
  parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] += bias_data[out_f];
  });
}

#define INSTANTIATE_DENSE(T)                                                                       \
  template void compute_dense_forward<T>(                                                          \
      const T *input_data, const T *weight_data, T *output_data, const size_t batch_size,          \
      const size_t input_features, const size_t output_features);                                  \
                                                                                                   \
  template void compute_weight_gradients<T>(                                                       \
      const T *input_data, const T *gradient_data, T *weight_grad_data, const size_t batch_size,   \
      const size_t input_features, const size_t output_features);                                  \
                                                                                                   \
  template void compute_input_gradients<T>(                                                        \
      const T *gradient_data, const T *weight_data, T *grad_input_data, const size_t batch_size,   \
      const size_t input_features, const size_t output_features);                                  \
                                                                                                   \
  template void compute_bias_gradients<T>(const T *current_grad_data, T *bias_gradient_data,       \
                                          const size_t batch_size, const size_t output_features);  \
                                                                                                   \
  template void add_bias_vector<T>(T * output_data, const T *bias_data, const size_t batch_size,   \
                                   const size_t output_features);
INSTANTIATE_DENSE(fp16)
INSTANTIATE_DENSE(float)
INSTANTIATE_DENSE(double)
#undef INSTANTIATE_DENSE
} // namespace dense
} // namespace cpu
} // namespace tnn
