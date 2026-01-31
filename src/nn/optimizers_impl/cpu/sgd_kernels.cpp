/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cpu/sgd_kernels.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace sgd {

template <typename T>
void update_sgd(T *params_data, const T *grads_data, const size_t size, const float learning_rate) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    params_data[i] -= learning_rate * static_cast<float>(grads_data[i]);
  });
}

template <typename T>
void update_sgd_momentum(T *params_data, const T *grads_data, T *velocity_data, const size_t size,
                         const float learning_rate, const float momentum) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    velocity_data[i] = momentum * static_cast<float>(velocity_data[i]) -
                       learning_rate * static_cast<float>(grads_data[i]);
    params_data[i] += velocity_data[i];
  });
}

#define INSTANTIATE_SGD(T)                                                                     \
  template void update_sgd<T>(T * params_data, const T *grads_data, const size_t size,         \
                              const float learning_rate);                                      \
  template void update_sgd_momentum<T>(T * params_data, const T *grads_data, T *velocity_data, \
                                       const size_t size, const float learning_rate,           \
                                       const float momentum);
INSTANTIATE_SGD(fp16)
INSTANTIATE_SGD(bf16)
INSTANTIATE_SGD(float)
INSTANTIATE_SGD(double)
#undef INSTANTIATE_SGD

}  // namespace sgd
}  // namespace cpu
}  // namespace tnn
