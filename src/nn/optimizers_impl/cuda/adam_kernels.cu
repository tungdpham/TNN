/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "cuda/error_handler.hpp"
#include "nn/optimizers_impl/cuda/adam_kernels.hpp"
#include "type/cuda/vectorized_types.hpp"
#include "type/type.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {
namespace adam {

template <typename T>
__global__ void update_adam_kernel(T* params_data, const T* grads_data, T* m_data, T* v_data,
                                   const size_t size, const float learning_rate, const float beta1,
                                   const float beta2, const float epsilon,
                                   const float bias_correction1, const float bias_correction2,
                                   const float weight_decay, const bool decouple_weight_decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size) return;

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
__global__ void update_adam_kernel_vec(T* params_data, const T* grads_data, T* m_data, T* v_data,
                                       const size_t size, const float learning_rate,
                                       const float beta1, const float beta2, const float epsilon,
                                       const float bias_correction1, const float bias_correction2,
                                       const float weight_decay, const bool decouple_weight_decay) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  if (idx >= size) return;
  T p_arr[vec_size], g_arr[vec_size], m_arr[vec_size], v_arr[vec_size];
  *reinterpret_cast<VecT*>(p_arr) = reinterpret_cast<const VecT*>(params_data)[idx / vec_size];
  *reinterpret_cast<VecT*>(g_arr) = reinterpret_cast<const VecT*>(grads_data)[idx / vec_size];
  *reinterpret_cast<VecT*>(m_arr) = reinterpret_cast<const VecT*>(m_data)[idx / vec_size];
  *reinterpret_cast<VecT*>(v_arr) = reinterpret_cast<const VecT*>(v_data)[idx / vec_size];
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    if (idx + i < size) {
      float grad = static_cast<float>(g_arr[i]);
      float param = static_cast<float>(p_arr[i]);
      float m = static_cast<float>(m_arr[i]);
      float v = static_cast<float>(v_arr[i]);

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

      m_arr[i] = static_cast<T>(m);
      v_arr[i] = static_cast<T>(v);
      p_arr[i] = static_cast<T>(param);
    }
  }

  reinterpret_cast<VecT*>(params_data)[idx / vec_size] = *reinterpret_cast<VecT*>(p_arr);
  reinterpret_cast<VecT*>(m_data)[idx / vec_size] = *reinterpret_cast<VecT*>(m_arr);
  reinterpret_cast<VecT*>(v_data)[idx / vec_size] = *reinterpret_cast<VecT*>(v_arr);
}

template <typename T>
void update_adam(T* params_data, const T* grads_data, T* m_data, T* v_data, const size_t size,
                 const float learning_rate, const float beta1, const float beta2,
                 const float epsilon, const float bias_correction1, const float bias_correction2,
                 const float weight_decay, const bool decouple_weight_decay, cudaStream_t stream) {
  const int threads_per_block = 256;
  constexpr size_t vec_size = VectoredTraits<T>::size;

  // Check if pointer is aligned to the vector type (usually 16 bytes for float4)
  bool is_aligned =
      (reinterpret_cast<uintptr_t>(params_data) % alignof(typename VectoredTraits<T>::type) == 0);

  if (is_aligned && size % vec_size == 0) {
    const int num_blocks_vec = (size / vec_size + threads_per_block - 1) / threads_per_block;
    update_adam_kernel_vec<<<num_blocks_vec, threads_per_block, 0, stream>>>(
        params_data, grads_data, m_data, v_data, size, learning_rate, beta1, beta2, epsilon,
        bias_correction1, bias_correction2, weight_decay, decouple_weight_decay);

  } else {
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    update_adam_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        params_data, grads_data, m_data, v_data, size, learning_rate, beta1, beta2, epsilon,
        bias_correction1, bias_correction2, weight_decay, decouple_weight_decay);
  }
  CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_ADAM_KERNELS(T)                                                         \
  template void update_adam<T>(                                                             \
      T * params_data, const T* grads_data, T* m_data, T* v_data, const size_t size,        \
      const float learning_rate, const float beta1, const float beta2, const float epsilon, \
      const float bias_correction1, const float bias_correction2, const float weight_decay, \
      const bool decouple_weight_decay, cudaStream_t stream);
INSTANTIATE_ADAM_KERNELS(fp16)
INSTANTIATE_ADAM_KERNELS(bf16)
INSTANTIATE_ADAM_KERNELS(float)
INSTANTIATE_ADAM_KERNELS(double)
#undef INSTANTIATE_ADAM_KERNELS

}  // namespace adam
}  // namespace cuda
}  // namespace tnn

#endif
