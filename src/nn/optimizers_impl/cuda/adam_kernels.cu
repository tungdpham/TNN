/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cuda/adam_kernels.hpp"
#include "type/type.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {
namespace adam {

template <typename T>
__global__ void update_adam_kernel(T *params_data, const T *grads_data, T *m_data, T *v_data,
                                   const size_t size, const float learning_rate, const float beta1,
                                   const float beta2, const float epsilon,
                                   const float bias_correction1, const float bias_correction2,
                                   const float weight_decay, const bool decouple_weight_decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size)
    return;

  float grad = static_cast<float>(grads_data[idx]);
  float param = static_cast<float>(params_data[idx]);
  float m = static_cast<float>(m_data[idx]);
  float v = static_cast<float>(v_data[idx]);

  m = beta1 * m + (1.0f - beta1) * grad;
  v = beta2 * v + (1.0f - beta2) * grad * grad;

  float m_hat = m / bias_correction1;
  float v_hat = v / bias_correction2;

  float update = (learning_rate * m_hat) / (sqrtf(v_hat) + epsilon);

  if (weight_decay > 0.0f) {
    if (decouple_weight_decay) {
      param -= weight_decay * learning_rate * param;
    } else {
      update += weight_decay * learning_rate * param;
    }
  }

  param -= update;

  m_data[idx] = static_cast<T>(m);
  v_data[idx] = static_cast<T>(v);
  params_data[idx] = static_cast<T>(param);
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

#define INSTANTIATE_ADAM_KERNELS(T)                                                                \
  template void update_adam<T>(                                                                    \
      T * params_data, const T *grads_data, T *m_data, T *v_data, const size_t size,               \
      const float learning_rate, const float beta1, const float beta2, const float epsilon,        \
      const float bias_correction1, const float bias_correction2, const float weight_decay,        \
      const bool decouple_weight_decay, cudaStream_t stream);
INSTANTIATE_ADAM_KERNELS(fp16)
INSTANTIATE_ADAM_KERNELS(bf16)
INSTANTIATE_ADAM_KERNELS(float)
INSTANTIATE_ADAM_KERNELS(double)
#undef INSTANTIATE_ADAM_KERNELS

} // namespace adam
} // namespace cuda
} // namespace tnn

#endif
