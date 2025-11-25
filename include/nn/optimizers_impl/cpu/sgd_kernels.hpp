#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace sgd {

// SGD without momentum: params -= learning_rate * grads
template <typename T>
void update_sgd(T *params_data, const T *grads_data, const size_t size, const float learning_rate);

// SGD with momentum: velocity = momentum * velocity - learning_rate * grads
//                    params += velocity
template <typename T>
void update_sgd_momentum(T *params_data, const T *grads_data, T *velocity_data, const size_t size,
                         const float learning_rate, const float momentum);

} // namespace sgd
} // namespace cpu
} // namespace tnn