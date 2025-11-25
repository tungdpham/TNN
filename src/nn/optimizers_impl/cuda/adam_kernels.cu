/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cuda/adam_kernels.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {
namespace adam {

// Fused Adam kernel that performs all operations in a single pass
template <typename T>
__global__ void update_adam_kernel(T *params_data, const T *grads_data, T *m_data, T *v_data,
                                   const size_t size, const float learning_rate, const float beta1,
                                   const float beta2, const float epsilon,
                                   const float bias_correction1, const float bias_correction2,
                                   const float weight_decay, const bool decouple_weight_decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    T grad = grads_data[idx];
    T param = params_data[idx];

    // Precompute constants
    const T one_minus_beta1 = static_cast<T>(1.0) - beta1;
    const T one_minus_beta2 = static_cast<T>(1.0) - beta2;

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
    T m = beta1 * m_data[idx] + one_minus_beta1 * grad;
    m_data[idx] = m;

    // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
    T v = beta2 * v_data[idx] + one_minus_beta2 * grad * grad;
    v_data[idx] = v;

    // Compute bias-corrected estimates
    T m_hat = m / bias_correction1;
    T v_hat = v / bias_correction2;

    // Compute update: lr * m_hat / (sqrt(v_hat) + epsilon)
    T update = learning_rate * m_hat / (sqrt(v_hat) + epsilon);

    // Apply weight decay
    if (weight_decay > 0.0f) {
      if (decouple_weight_decay) {
        // AdamW: decoupled weight decay applied directly to parameters
        param -= weight_decay * learning_rate * param;
      } else {
        // Adam with L2 regularization: add to update
        update += weight_decay * learning_rate * param;
      }
    }

    // Update parameters
    params_data[idx] = param - update;
  }
}

template <typename T>
void update_adam(T *params_data, const T *grads_data, T *m_data, T *v_data, const size_t size,
                 const float learning_rate, const float beta1, const float beta2,
                 const float epsilon, const float bias_correction1, const float bias_correction2,
                 const float weight_decay, const bool decouple_weight_decay, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  update_adam_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      params_data, grads_data, m_data, v_data, size, learning_rate, beta1, beta2, epsilon,
      bias_correction1, bias_correction2, weight_decay, decouple_weight_decay);
}

// Explicit template instantiations
template void update_adam<float>(float *params_data, const float *grads_data, float *m_data,
                                 float *v_data, const size_t size, const float learning_rate,
                                 const float beta1, const float beta2, const float epsilon,
                                 const float bias_correction1, const float bias_correction2,
                                 const float weight_decay, const bool decouple_weight_decay,
                                 cudaStream_t stream);
template void update_adam<double>(double *params_data, const double *grads_data, double *m_data,
                                  double *v_data, const size_t size, const float learning_rate,
                                  const float beta1, const float beta2, const float epsilon,
                                  const float bias_correction1, const float bias_correction2,
                                  const float weight_decay, const bool decouple_weight_decay,
                                  cudaStream_t stream);

} // namespace adam
} // namespace cuda
} // namespace tnn

#endif
