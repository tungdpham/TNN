/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/layer_norm_ops.hpp"

#include <cmath>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace layer_norm {

namespace {

// atomicAdd for double is not available on very old GPUs.
__device__ inline double atomicAddCompat(double *address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

template <typename T> __device__ inline T atomicAddT(T *address, T val) {
  return atomicAdd(address, val);
}

template <> __device__ inline double atomicAddT<double>(double *address, double val) {
  return atomicAddCompat(address, val);
}

template <typename T>
__global__ void layer_norm_forward_kernel(const T *input, T *output, const T *gamma, const T *beta,
                                          size_t channels, T epsilon) {
  const size_t n = static_cast<size_t>(blockIdx.x);

  const T *x = input + n * channels;
  T *y = output + n * channels;

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += x[c];
  }
  const T mean = sum / static_cast<T>(channels);

  T sq_sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    const T diff = x[c] - mean;
    sq_sum += diff * diff;
  }
  const T var = sq_sum / static_cast<T>(channels);
  const T inv_std = T(1) / sqrt(var + epsilon);

  for (size_t c = 0; c < channels; ++c) {
    const T norm = (x[c] - mean) * inv_std;
    const T g = gamma ? gamma[c] : T(1);
    const T b = beta ? beta[c] : T(0);
    y[c] = g * norm + b;
  }
}

template <typename T>
__global__ void layer_norm_backward_kernel(const T *grad_output, const T *input, const T *gamma,
                                           T *grad_input, T *grad_gamma, T *grad_beta,
                                           size_t channels, T epsilon) {
  const size_t n = static_cast<size_t>(blockIdx.x);

  const T *x = input + n * channels;
  const T *go = grad_output + n * channels;
  T *gi = grad_input ? (grad_input + n * channels) : nullptr;

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += x[c];
  }
  const T mean = sum / static_cast<T>(channels);

  T sq_sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    const T diff = x[c] - mean;
    sq_sum += diff * diff;
  }
  const T var = sq_sum / static_cast<T>(channels);
  const T inv_std = T(1) / sqrt(var + epsilon);

  // For input gradient
  T sum_dl_dnorm = T(0);
  T sum_dl_dnorm_x_hat = T(0);

  for (size_t c = 0; c < channels; ++c) {
    const T g = gamma ? gamma[c] : T(1);
    const T dl_dnorm = go[c] * g;
    const T x_hat = (x[c] - mean) * inv_std;
    sum_dl_dnorm += dl_dnorm;
    sum_dl_dnorm_x_hat += dl_dnorm * x_hat;

    if (grad_gamma) {
      atomicAddT(&grad_gamma[c], go[c] * x_hat);
    }
    if (grad_beta) {
      atomicAddT(&grad_beta[c], go[c]);
    }
  }

  if (gi) {
    const T mean_dl_dnorm = sum_dl_dnorm / static_cast<T>(channels);
    const T mean_dl_dnorm_x_hat = sum_dl_dnorm_x_hat / static_cast<T>(channels);

    for (size_t c = 0; c < channels; ++c) {
      const T g = gamma ? gamma[c] : T(1);
      const T dl_dnorm = go[c] * g;
      const T x_hat = (x[c] - mean) * inv_std;
      gi[c] = inv_std * (dl_dnorm - mean_dl_dnorm - x_hat * mean_dl_dnorm_x_hat);
    }
  }
}

} // namespace

template <typename T>
void layer_norm_forward(const T *input, T *output, const T *gamma, const T *beta, size_t batch_size,
                        size_t channels, T epsilon, cudaStream_t stream) {
  if (batch_size == 0 || channels == 0) {
    return;
  }
  dim3 blocks(static_cast<unsigned int>(batch_size));
  dim3 threads(1);
  layer_norm_forward_kernel<T>
      <<<blocks, threads, 0, stream>>>(input, output, gamma, beta, channels, epsilon);
}

template <typename T>
void layer_norm_backward(const T *grad_output, const T *input, const T *gamma, T *grad_input,
                         T *grad_gamma, T *grad_beta, size_t batch_size, size_t channels, T epsilon,
                         cudaStream_t stream) {
  if (batch_size == 0 || channels == 0) {
    return;
  }
  dim3 blocks(static_cast<unsigned int>(batch_size));
  dim3 threads(1);
  layer_norm_backward_kernel<T><<<blocks, threads, 0, stream>>>(
      grad_output, input, gamma, grad_input, grad_gamma, grad_beta, channels, epsilon);
}

template void layer_norm_forward<float>(const float *input, float *output, const float *gamma,
                                        const float *beta, size_t batch_size, size_t channels,
                                        float epsilon, cudaStream_t stream);
template void layer_norm_backward<float>(const float *grad_output, const float *input,
                                         const float *gamma, float *grad_input, float *grad_gamma,
                                         float *grad_beta, size_t batch_size, size_t channels,
                                         float epsilon, cudaStream_t stream);

template void layer_norm_forward<double>(const double *input, double *output, const double *gamma,
                                         const double *beta, size_t batch_size, size_t channels,
                                         double epsilon, cudaStream_t stream);
template void layer_norm_backward<double>(const double *grad_output, const double *input,
                                          const double *gamma, double *grad_input,
                                          double *grad_gamma, double *grad_beta, size_t batch_size,
                                          size_t channels, double epsilon, cudaStream_t stream);

} // namespace layer_norm
} // namespace cuda
} // namespace tnn
