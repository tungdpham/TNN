/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <ctime>

#include "nn/layers_impl/cuda/dropout_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace dropout {
#define BLOCK_SIZE 256

template <typename T>
struct VectorType;
template <>
struct VectorType<float> {
  using type = float4;
};
template <>
struct VectorType<double> {
  using type = double2;
};

template <typename T>
__global__ void run_forward_kernel_vectorized(const T* __restrict__ input_data,
                                              T* __restrict__ output_data,
                                              bool* __restrict__ mask_data, size_t n_elements,
                                              T dropout_rate, T scale, unsigned long long seed) {
  using VecT = typename VectorType<T>::type;
  constexpr size_t vec_width = sizeof(VecT) / sizeof(T);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input_data);
  VecT* output_vec = reinterpret_cast<VecT*>(output_data);

  size_t n_vectors = n_elements / vec_width;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < n_vectors; i += stride) {
    VecT in_val = input_vec[i];
    VecT out_val;

    float4 rand_vals = curand_uniform4(&state);
    size_t base = i * vec_width;

    if (rand_vals.x < dropout_rate) {
      mask_data[base] = false;
      out_val.x = 0.0f;
    } else {
      mask_data[base] = true;
      out_val.x = in_val.x * scale;
    }

    if (rand_vals.y < dropout_rate) {
      mask_data[base + 1] = false;
      out_val.y = 0.0f;
    } else {
      mask_data[base + 1] = true;
      out_val.y = in_val.y * scale;
    }

    if (rand_vals.z < dropout_rate) {
      mask_data[base + 2] = false;
      out_val.z = 0.0f;
    } else {
      mask_data[base + 2] = true;
      out_val.z = in_val.z * scale;
    }

    if (rand_vals.w < dropout_rate) {
      mask_data[base + 3] = false;
      out_val.w = 0.0f;
    } else {
      mask_data[base + 3] = true;
      out_val.w = in_val.w * scale;
    }

    output_vec[i] = out_val;
  }

  size_t tail_start = n_vectors * vec_width;
  for (size_t i = tail_start + idx; i < n_elements; i += stride) {
    float r = curand_uniform(&state);
    if (r < dropout_rate) {
      mask_data[i] = false;
      output_data[i] = T(0);
    } else {
      mask_data[i] = true;
      output_data[i] = input_data[i] * scale;
    }
  }
}

template <typename T>
__global__ void run_forward_kernel(const T* input_data, T* output_data, bool* mask_data,
                                   size_t batch_size, size_t channels, size_t spatial_size,
                                   T dropout_rate, T scale, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_size * channels * spatial_size;
  int stride = blockDim.x * gridDim.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < total_elements; i += stride) {
    float rand_val = curand_uniform(&state);

    if (rand_val < static_cast<float>(dropout_rate)) {
      mask_data[i] = false;
      output_data[i] = T(0);
    } else {
      mask_data[i] = true;
      output_data[i] = input_data[i] * scale;
    }
  }
}

template <typename T>
void run_forward(const T* input_data, T* output_data, bool* mask_data, size_t batch_size,
                 size_t channels, size_t spatial_size, T dropout_rate, cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;

  int threads = BLOCK_SIZE;
  int blocks = (total_elements + threads - 1) / threads;
  blocks = std::min(blocks, 4096);

  T scale = T(1) / (T(1) - dropout_rate);

  unsigned long long seed = static_cast<unsigned long long>(clock()) +
                            static_cast<unsigned long long>(std::time(nullptr));

  if constexpr (std::is_same<T, float>::value) {
    run_forward_kernel_vectorized<float><<<blocks, threads, 0, stream>>>(
        input_data, output_data, mask_data, total_elements, dropout_rate, scale, seed);
  } else {
    run_forward_kernel<<<blocks, threads, 0, stream>>>(input_data, output_data, mask_data,
                                                       batch_size, channels, spatial_size,
                                                       dropout_rate, scale, seed);
  }
}

template <typename T>
__global__ void run_backward_kernel(const T* __restrict__ grad_output_data,
                                    T* __restrict__ grad_input_data,
                                    const bool* __restrict__ mask_data, size_t total_elements,
                                    T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total_elements; i += stride) {
    grad_input_data[i] = mask_data[i] ? grad_output_data[i] * scale : T(0);
  }
}

template <typename T>
void run_backward(const T* grad_output_data, T* grad_input_data, const bool* mask_data,
                  size_t batch_size, size_t channels, size_t spatial_size, T scale,
                  cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;

  int threads = BLOCK_SIZE;
  int blocks = (total_elements + threads - 1) / threads;
  blocks = std::min(blocks, 4096);

  run_backward_kernel<T><<<blocks, threads, 0, stream>>>(grad_output_data, grad_input_data,
                                                         mask_data, total_elements, scale);
}

#define INSTANTIATE_DROPOUT(T)                                                             \
  template void run_forward<T>(const T* input_data, T* output_data, bool* mask_data,       \
                               size_t batch_size, size_t channels, size_t spatial_size,    \
                               T dropout_rate, cudaStream_t stream);                       \
  template void run_backward<T>(const T* grad_output_data, T* grad_input_data,             \
                                const bool* mask_data, size_t batch_size, size_t channels, \
                                size_t spatial_size, T scale, cudaStream_t stream);
INSTANTIATE_DROPOUT(fp16)
INSTANTIATE_DROPOUT(bf16)
INSTANTIATE_DROPOUT(float)
INSTANTIATE_DROPOUT(double)
#undef INSTANTIATE_DROPOUT

}  // namespace dropout
}  // namespace cuda
}  // namespace tnn