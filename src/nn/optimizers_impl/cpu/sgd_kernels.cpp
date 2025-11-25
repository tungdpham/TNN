/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cpu/sgd_kernels.hpp"

#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {
namespace sgd {

template <typename T>
void update_sgd(T *params_data, const T *grads_data, const size_t size, const float learning_rate) {
  // params -= learning_rate * grads
  parallel_for<size_t>(0, size, [&](size_t i) { params_data[i] -= learning_rate * grads_data[i]; });
}

template <typename T>
void update_sgd_momentum(T *params_data, const T *grads_data, T *velocity_data, const size_t size,
                         const float learning_rate, const float momentum) {
  // velocity = momentum * velocity - learning_rate * grads
  // params += velocity
  parallel_for<size_t>(0, size, [&](size_t i) {
    velocity_data[i] = momentum * velocity_data[i] - learning_rate * grads_data[i];
    params_data[i] += velocity_data[i];
  });
}

// Explicit template instantiations
template void update_sgd<float>(float *params_data, const float *grads_data, const size_t size,
                                const float learning_rate);
template void update_sgd<double>(double *params_data, const double *grads_data, const size_t size,
                                 const float learning_rate);

template void update_sgd_momentum<float>(float *params_data, const float *grads_data,
                                         float *velocity_data, const size_t size,
                                         const float learning_rate, const float momentum);
template void update_sgd_momentum<double>(double *params_data, const double *grads_data,
                                          double *velocity_data, const size_t size,
                                          const float learning_rate, const float momentum);

} // namespace sgd
} // namespace cpu
} // namespace tnn
