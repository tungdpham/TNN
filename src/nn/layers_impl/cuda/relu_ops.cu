/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "nn/layers_impl/cuda/relu_ops.hpp"
#include "type/cuda/vectorized_types.hpp"

namespace tnn {
namespace cuda {
namespace relu {

template <typename T>
__global__ void relu_forward_with_mask_vec_kernel(const T* input, T* output, uint8_t* mask,
                                                  size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  VecT in_val = input_vec[idx];
  VecT out_val;

  const T* in_ptr = reinterpret_cast<const T*>(&in_val);
  T* out_ptr = reinterpret_cast<T*>(&out_val);

  T zero = static_cast<T>(0);
  uint8_t mask_vals[VecSize];

  for (int i = 0; i < VecSize; ++i) {
    bool is_positive = in_ptr[i] > zero;
    out_ptr[i] = is_positive ? in_ptr[i] : zero;
    mask_vals[i] = is_positive ? 1 : 0;
  }

  output_vec[idx] = out_val;

  size_t mask_base = idx * VecSize;
  for (int i = 0; i < VecSize; ++i) {
    mask[mask_base + i] = mask_vals[i];
  }
}

template <typename T>
__global__ void relu_forward_with_mask_tail_kernel(const T* input, T* output, uint8_t* mask,
                                                   size_t offset, size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  T val = input[pos];
  T zero = static_cast<T>(0);
  bool is_positive = val > zero;
  output[pos] = is_positive ? val : zero;
  mask[pos] = is_positive ? 1 : 0;
}

template <typename T>
void relu_forward_with_mask(const T* input_data, T* output_data, uint8_t* mask_data,
                            size_t num_elements, cudaStream_t stream) {
  if (num_elements == 0) return;

  constexpr int VecSize = VectoredTraits<T>::size;
  constexpr int threads = 256;

  size_t n_vectors = num_elements / VecSize;
  if (n_vectors > 0) {
    int blocks = (n_vectors + threads - 1) / threads;
    relu_forward_with_mask_vec_kernel<T>
        <<<blocks, threads, 0, stream>>>(input_data, output_data, mask_data, n_vectors);
  }

  size_t tail_offset = n_vectors * VecSize;
  size_t tail_elements = num_elements - tail_offset;
  if (tail_elements > 0) {
    int blocks = (tail_elements + threads - 1) / threads;
    relu_forward_with_mask_tail_kernel<T><<<blocks, threads, 0, stream>>>(
        input_data, output_data, mask_data, tail_offset, tail_elements);
  }
}

template <typename T>
__global__ void relu_backward_with_mask_vec_kernel(const T* grad_output, T* grad_input,
                                                   const uint8_t* mask, size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* grad_out_vec = reinterpret_cast<const VecT*>(grad_output);
  VecT* grad_in_vec = reinterpret_cast<VecT*>(grad_input);

  VecT grad_out_val = grad_out_vec[idx];
  VecT grad_in_val;

  const T* grad_out_ptr = reinterpret_cast<const T*>(&grad_out_val);
  T* grad_in_ptr = reinterpret_cast<T*>(&grad_in_val);

  size_t mask_base = idx * VecSize;

  for (int i = 0; i < VecSize; ++i) {
    uint8_t m = mask[mask_base + i];
    grad_in_ptr[i] = grad_out_ptr[i] * static_cast<T>(m);
  }

  grad_in_vec[idx] = grad_in_val;
}

template <typename T>
__global__ void relu_backward_with_mask_tail_kernel(const T* grad_output, T* grad_input,
                                                    const uint8_t* mask, size_t offset,
                                                    size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  grad_input[pos] = grad_output[pos] * static_cast<T>(mask[pos]);
}

template <typename T>
void relu_backward_with_mask(const T* grad_output_data, T* grad_input_data,
                             const uint8_t* mask_data, size_t num_elements, cudaStream_t stream) {
  if (num_elements == 0) return;

  constexpr int VecSize = VectoredTraits<T>::size;
  constexpr int threads = 256;

  size_t n_vectors = num_elements / VecSize;
  if (n_vectors > 0) {
    int blocks = (n_vectors + threads - 1) / threads;
    relu_backward_with_mask_vec_kernel<T>
        <<<blocks, threads, 0, stream>>>(grad_output_data, grad_input_data, mask_data, n_vectors);
  }

  size_t tail_offset = n_vectors * VecSize;
  size_t tail_elements = num_elements - tail_offset;
  if (tail_elements > 0) {
    int blocks = (tail_elements + threads - 1) / threads;
    relu_backward_with_mask_tail_kernel<T><<<blocks, threads, 0, stream>>>(
        grad_output_data, grad_input_data, mask_data, tail_offset, tail_elements);
  }
}

#define INSTANTIATE(T)                                                                             \
  template void relu_forward_with_mask<T>(const T* input_data, T* output_data, uint8_t* mask_data, \
                                          size_t num_elements, cudaStream_t stream);               \
                                                                                                   \
  template void relu_backward_with_mask<T>(const T* grad_output_data, T* grad_input_data,          \
                                           const uint8_t* mask_data, size_t num_elements,          \
                                           cudaStream_t stream);

#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace relu
}  // namespace cuda
}  // namespace tnn

#endif
