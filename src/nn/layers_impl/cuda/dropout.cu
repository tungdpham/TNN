/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/dropout_ops.hpp"

#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tnn {
namespace cuda {

#define BLOCK_SIZE 256

template <typename T> struct VectorType;
template <> struct VectorType<float> {
  using type = float4;
};
template <> struct VectorType<double> {
  using type = double2;
};

template <typename T>
__global__ void compute_dropout_forward_kernel_vectorized(const T *__restrict__ input_data,
                                                          T *__restrict__ output_data,
                                                          T *__restrict__ mask_data,
                                                          size_t n_elements, T dropout_rate,
                                                          T scale, unsigned long long seed) {
  using VecT = typename VectorType<T>::type;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  const VecT *input_vec = reinterpret_cast<const VecT *>(input_data);
  VecT *output_vec = reinterpret_cast<VecT *>(output_data);
  VecT *mask_vec = reinterpret_cast<VecT *>(mask_data);

  size_t n_vectors = n_elements / (sizeof(VecT) / sizeof(T));

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < n_vectors; i += stride) {
    VecT in_val = input_vec[i];
    VecT out_val;
    VecT mask_val;

    float4 rand_vals = curand_uniform4(&state);

    if (rand_vals.x < dropout_rate) {
      mask_val.x = 0.0f;
      out_val.x = 0.0f;
    } else {
      mask_val.x = scale;
      out_val.x = in_val.x * scale;
    }

    if (rand_vals.y < dropout_rate) {
      mask_val.y = 0.0f;
      out_val.y = 0.0f;
    } else {
      mask_val.y = scale;
      out_val.y = in_val.y * scale;
    }

    if (rand_vals.z < dropout_rate) {
      mask_val.z = 0.0f;
      out_val.z = 0.0f;
    } else {
      mask_val.z = scale;
      out_val.z = in_val.z * scale;
    }

    if (rand_vals.w < dropout_rate) {
      mask_val.w = 0.0f;
      out_val.w = 0.0f;
    } else {
      mask_val.w = scale;
      out_val.w = in_val.w * scale;
    }

    output_vec[i] = out_val;
    mask_vec[i] = mask_val;
  }

  size_t tail_start = n_vectors * (sizeof(VecT) / sizeof(T));
  for (size_t i = tail_start + idx; i < n_elements; i += stride) {
    float r = curand_uniform(&state);
    if (r < dropout_rate) {
      mask_data[i] = T(0);
      output_data[i] = T(0);
    } else {
      mask_data[i] = scale;
      output_data[i] = input_data[i] * scale;
    }
  }
}

template <typename T>
__global__ void compute_dropout_forward_kernel(const T *input_data, T *output_data, T *mask_data,
                                               size_t batch_size, size_t channels,
                                               size_t spatial_size, T dropout_rate, T scale,
                                               unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_size * channels * spatial_size;
  int stride = blockDim.x * gridDim.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < total_elements; i += stride) {
    float rand_val = curand_uniform(&state);

    if (rand_val < dropout_rate) {
      mask_data[i] = T(0);
      output_data[i] = T(0);
    } else {
      mask_data[i] = scale;
      output_data[i] = input_data[i] * scale;
    }
  }
}

template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, T *mask_data, size_t batch_size,
                             size_t channels, size_t spatial_size, T dropout_rate,
                             cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;

  int threads = BLOCK_SIZE;
  int blocks = (total_elements + threads - 1) / threads;
  blocks = std::min(blocks, 4096);

  T scale = T(1) / (T(1) - dropout_rate);

  unsigned long long seed = static_cast<unsigned long long>(clock()) +
                            static_cast<unsigned long long>(std::time(nullptr));

  if constexpr (std::is_same<T, float>::value) {
    compute_dropout_forward_kernel_vectorized<float><<<blocks, threads, 0, stream>>>(
        input_data, output_data, mask_data, total_elements, dropout_rate, scale, seed);
  } else {

    compute_dropout_forward_kernel<<<blocks, threads, 0, stream>>>(
        input_data, output_data, mask_data, batch_size, channels, spatial_size, dropout_rate, scale,
        seed);
  }
}

template void compute_dropout_forward<float>(const float *input_data, float *output_data,
                                             float *mask_data, size_t batch_size, size_t channels,
                                             size_t spatial_size, float dropout_rate,
                                             cudaStream_t stream);
template void compute_dropout_forward<double>(const double *input_data, double *output_data,
                                              double *mask_data, size_t batch_size, size_t channels,
                                              size_t spatial_size, double dropout_rate,
                                              cudaStream_t stream);

} // namespace cuda
} // namespace tnn