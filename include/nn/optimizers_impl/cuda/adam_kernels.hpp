#pragma once

#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace adam {

// Adam update step combining all operations:
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// params -= lr * m_hat / (sqrt(v_hat) + epsilon)
// Optional weight decay: params -= lr * weight_decay * params (AdamW) or add to gradient (Adam)
template <typename T>
void update_adam(T *params_data, const T *grads_data, T *m_data, T *v_data, const size_t size,
                 const float learning_rate, const float beta1, const float beta2,
                 const float epsilon, const float bias_correction1, const float bias_correction2,
                 const float weight_decay, const bool decouple_weight_decay, cudaStream_t stream);

} // namespace adam
} // namespace cuda
} // namespace tnn

#endif