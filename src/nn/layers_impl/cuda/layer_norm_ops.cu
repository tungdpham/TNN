/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/layer_norm_ops.hpp"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace layer_norm {

template <typename T>
__global__ void layer_norm_forward_impl(const T *input, T *output, const T *gamma, const T *beta,
                                        size_t channels, size_t spatial_size, T epsilon,
                                        size_t total_spatial_items) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_spatial_items)
    return;

  size_t s = idx % spatial_size;
  size_t n = idx / spatial_size;
  size_t base_offset = n * channels * spatial_size + s;
  size_t stride = spatial_size;

  T sum = 0;
  for (size_t c = 0; c < channels; ++c) {
    sum += input[base_offset + c * stride];
  }
  T mean = sum / static_cast<T>(channels);

  T sq_sum = 0;
  for (size_t c = 0; c < channels; ++c) {
    T val = input[base_offset + c * stride];
    T diffuse = val - mean;
    sq_sum += diffuse * diffuse;
  }
  T var = sq_sum / static_cast<T>(channels);
  T inv_std = rsqrt(var + epsilon);

  for (size_t c = 0; c < channels; ++c) {
    size_t addr = base_offset + c * stride;
    T val = input[addr];
    T norm = (val - mean) * inv_std;
    T g = gamma ? gamma[c] : static_cast<T>(1);
    T b = beta ? beta[c] : static_cast<T>(0);
    output[addr] = g * norm + b;
  }
}

template <typename T>
void layer_norm_forward(const T *input, T *output, const T *gamma, const T *beta, size_t batch_size,
                        size_t channels, size_t spatial_size, T epsilon, cudaStream_t stream) {
  size_t total_spatial_items = batch_size * spatial_size;
  int threads = 256;
  int blocks = (total_spatial_items + threads - 1) / threads;
  layer_norm_forward_impl<T><<<blocks, threads, 0, stream>>>(
      input, output, gamma, beta, channels, spatial_size, epsilon, total_spatial_items);
}

template <typename T>
__global__ void layer_norm_backward_input_impl(const T *grad_output, const T *input, const T *gamma,
                                               T *grad_input, size_t channels, size_t spatial_size,
                                               T epsilon, size_t total_spatial_items) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_spatial_items)
    return;

  size_t s = idx % spatial_size;
  size_t n = idx / spatial_size;
  size_t base_offset = n * channels * spatial_size + s;
  size_t stride = spatial_size;

  T sum = 0;
  for (size_t c = 0; c < channels; ++c) {
    sum += input[base_offset + c * stride];
  }
  T mean = sum / static_cast<T>(channels);

  T sq_sum = 0;
  for (size_t c = 0; c < channels; ++c) {
    T val = input[base_offset + c * stride];
    sq_sum += (val - mean) * (val - mean);
  }
  T var = sq_sum / static_cast<T>(channels);
  T inv_std = rsqrt(var + epsilon);

  T sum_dl_dnorm = 0;
  T sum_dl_dnorm_x_hat = 0;

  for (size_t c = 0; c < channels; ++c) {
    size_t addr = base_offset + c * stride;
    T val = input[addr];
    T go = grad_output[addr];
    T g = gamma ? gamma[c] : static_cast<T>(1);
    T dl_dnorm = go * g;
    T x_hat = (val - mean) * inv_std;

    sum_dl_dnorm += dl_dnorm;
    sum_dl_dnorm_x_hat += dl_dnorm * x_hat;
  }

  T mean_dl_dnorm = sum_dl_dnorm / static_cast<T>(channels);
  T mean_dl_dnorm_x_hat = sum_dl_dnorm_x_hat / static_cast<T>(channels);

  for (size_t c = 0; c < channels; ++c) {
    size_t addr = base_offset + c * stride;
    T val = input[addr];
    T go = grad_output[addr];
    T g = gamma ? gamma[c] : static_cast<T>(1);
    T dl_dnorm = go * g;
    T x_hat = (val - mean) * inv_std;

    T dx = inv_std * (dl_dnorm - mean_dl_dnorm - x_hat * mean_dl_dnorm_x_hat);
    grad_input[addr] = dx;
  }
}

template <typename T>
__global__ void layer_norm_backward_params_impl(const T *grad_output, const T *input, T *grad_gamma,
                                                T *grad_beta, size_t channels, size_t spatial_size,
                                                T epsilon, size_t total_spatial_items) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_spatial_items)
    return;

  size_t s = idx % spatial_size;
  size_t n = idx / spatial_size;
  size_t base_offset = n * channels * spatial_size + s;
  size_t stride = spatial_size;

  T sum = 0;
  for (size_t c = 0; c < channels; ++c) {
    sum += input[base_offset + c * stride];
  }
  T mean = sum / static_cast<T>(channels);
  T sq_sum = 0;
  for (size_t c = 0; c < channels; ++c) {
    T val = input[base_offset + c * stride];
    sq_sum += (val - mean) * (val - mean);
  }
  T var = sq_sum / static_cast<T>(channels);
  T inv_std = rsqrt(var + epsilon);

  for (size_t c = 0; c < channels; ++c) {
    size_t addr = base_offset + c * stride;
    T go = grad_output[addr];
    T val = input[addr];
    T x_hat = (val - mean) * inv_std;

    if (grad_gamma)
      atomicAdd(&grad_gamma[c], go * x_hat);
    if (grad_beta)
      atomicAdd(&grad_beta[c], go);
  }
}

template <typename T>
void layer_norm_backward(const T *grad_output, const T *input, const T *gamma, T *grad_input,
                         T *grad_gamma, T *grad_beta, size_t batch_size, size_t channels,
                         size_t spatial_size, T epsilon, cudaStream_t stream) {
  size_t total_spatial_items = batch_size * spatial_size;
  int threads = 256;
  int blocks = (total_spatial_items + threads - 1) / threads;

  if (grad_input) {
    layer_norm_backward_input_impl<T>
        <<<blocks, threads, 0, stream>>>(grad_output, input, gamma, grad_input, channels,
                                         spatial_size, epsilon, total_spatial_items);
  }

  if (grad_gamma || grad_beta) {
    layer_norm_backward_params_impl<T>
        <<<blocks, threads, 0, stream>>>(grad_output, input, grad_gamma, grad_beta, channels,
                                         spatial_size, epsilon, total_spatial_items);
  }
}

template void layer_norm_forward<float>(const float *input, float *output, const float *gamma,
                                        const float *beta, size_t batch_size, size_t channels,
                                        size_t spatial_size, float epsilon, cudaStream_t stream);
template void layer_norm_backward<float>(const float *grad_output, const float *input,
                                         const float *gamma, float *grad_input, float *grad_gamma,
                                         float *grad_beta, size_t batch_size, size_t channels,
                                         size_t spatial_size, float epsilon, cudaStream_t stream);

template void layer_norm_forward<double>(const double *input, double *output, const double *gamma,
                                         const double *beta, size_t batch_size, size_t channels,
                                         size_t spatial_size, double epsilon, cudaStream_t stream);
template void layer_norm_backward<double>(const double *grad_output, const double *input,
                                          const double *gamma, double *grad_input,
                                          double *grad_gamma, double *grad_beta, size_t batch_size,
                                          size_t channels, size_t spatial_size, double epsilon,
                                          cudaStream_t stream);

} // namespace layer_norm
} // namespace cuda
} // namespace tnn
