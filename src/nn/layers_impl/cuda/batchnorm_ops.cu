/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/batchnorm_ops.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace batchnorm {
template <typename T>
__global__ void compute_channel_mean_kernel(const T *input_data, T *mean_data, size_t batch_size,
                                            size_t channels, size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T sum = T(0);
  for (size_t n = 0; n < batch_size; ++n) {
    const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      sum += batch_channel_ptr[i];
    }
  }

  mean_data[c] = sum * inv_total;
}

template <typename T>
__global__ void compute_channel_variance_kernel(const T *input_data, const T *mean_data,
                                                T *var_data, size_t batch_size, size_t channels,
                                                size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);
  const T mean_val = mean_data[c];
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T sum_sq = T(0);
  for (size_t n = 0; n < batch_size; ++n) {
    const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      T diff = batch_channel_ptr[i] - mean_val;
      sum_sq += diff * diff;
    }
  }

  var_data[c] = sum_sq * inv_total;
}

template <typename T>
__global__ void normalize_and_scale_kernel(const T *input_data, const T *mean_data,
                                           const T *std_data, const T *gamma_data,
                                           const T *beta_data, T *output_data, T *normalized_data,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  const T mean_val = mean_data[c];
  const T std_val = std_data[c];
  const T inv_std = T(1) / std_val;

  T input_val = input_data[idx];
  T normalized_val = (input_val - mean_val) * inv_std;
  normalized_data[idx] = normalized_val;

  if (affine) {
    const T gamma_val = gamma_data[c];
    const T beta_val = beta_data[c];
    output_data[idx] = gamma_val * normalized_val + beta_val;
  } else {
    output_data[idx] = normalized_val;
  }
}

template <typename T>
__global__ void compute_affine_gradients_kernel(const T *gradient_data, const T *normalized_data,
                                                T *gamma_grad, T *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T gamma_sum = T(0);
  T beta_sum = T(0);

  for (size_t n = 0; n < batch_size; ++n) {
    const size_t base_idx = n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      size_t idx = base_idx + i;
      gamma_sum += gradient_data[idx] * normalized_data[idx];
      beta_sum += gradient_data[idx];
    }
  }

  atomicAdd(&gamma_grad[c], gamma_sum);
  atomicAdd(&beta_grad[c], beta_sum);
}

template <typename T>
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_channel_mean_kernel<<<num_blocks, threads_per_block>>>(input_data, mean_data, batch_size,
                                                                 channels, spatial_size);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_channel_variance_kernel<<<num_blocks, threads_per_block>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
  cudaDeviceSynchronize();
}

template <typename T>
void normalize_and_scale_optimized(const T *input_data, const T *mean_data, const T *std_data,
                                   const T *gamma_data, const T *beta_data, T *output_data,
                                   T *normalized_data, size_t batch_size, size_t channels,
                                   size_t spatial_size, bool affine) {
  int total_elements = batch_size * channels * spatial_size;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  normalize_and_scale_kernel<<<num_blocks, threads_per_block>>>(
      input_data, mean_data, std_data, gamma_data, beta_data, output_data, normalized_data,
      batch_size, channels, spatial_size, affine);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_affine_gradients_optimized(const T *gradient_data, const T *normalized_data,
                                        T *gamma_grad, T *beta_grad, size_t batch_size,
                                        size_t channels, size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_affine_gradients_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, normalized_data, gamma_grad, beta_grad, batch_size, channels, spatial_size);
  cudaDeviceSynchronize();
}

// Explicit template instantiations
template void compute_channel_mean<float>(const float *input_data, float *mean_data,
                                          size_t batch_size, size_t channels, size_t spatial_size);
template void compute_channel_mean<double>(const double *input_data, double *mean_data,
                                           size_t batch_size, size_t channels, size_t spatial_size);

template void compute_channel_variance<float>(const float *input_data, const float *mean_data,
                                              float *var_data, size_t batch_size, size_t channels,
                                              size_t spatial_size);
template void compute_channel_variance<double>(const double *input_data, const double *mean_data,
                                               double *var_data, size_t batch_size, size_t channels,
                                               size_t spatial_size);

template void normalize_and_scale_optimized<float>(const float *input_data, const float *mean_data,
                                                   const float *std_data, const float *gamma_data,
                                                   const float *beta_data, float *output_data,
                                                   float *normalized_data, size_t batch_size,
                                                   size_t channels, size_t spatial_size,
                                                   bool affine);
template void normalize_and_scale_optimized<double>(
    const double *input_data, const double *mean_data, const double *std_data,
    const double *gamma_data, const double *beta_data, double *output_data, double *normalized_data,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine);

template void compute_affine_gradients_optimized<float>(const float *gradient_data,
                                                        const float *normalized_data,
                                                        float *gamma_grad, float *beta_grad,
                                                        size_t batch_size, size_t channels,
                                                        size_t spatial_size);
template void compute_affine_gradients_optimized<double>(const double *gradient_data,
                                                         const double *normalized_data,
                                                         double *gamma_grad, double *beta_grad,
                                                         size_t batch_size, size_t channels,
                                                         size_t spatial_size);

// New kernels

template <typename T>
__global__ void compute_batch_std_kernel(const T *batch_var_data, T *batch_std_data,
                                         size_t channels, T epsilon) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    batch_std_data[c] = sqrt(batch_var_data[c] + epsilon);
  }
}

template <typename T>
__global__ void update_running_stats_kernel(T *running_mean_data, T *running_var_data,
                                            const T *batch_mean_data, const T *batch_var_data,
                                            size_t channels, T momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    running_mean_data[c] = (T(1) - momentum) * running_mean_data[c] + momentum * batch_mean_data[c];
    running_var_data[c] = (T(1) - momentum) * running_var_data[c] + momentum * batch_var_data[c];
  }
}

template <typename T>
__global__ void compute_inference_output_kernel(const T *input_data, const T *running_mean_data,
                                                const T *running_var_data, const T *gamma_data,
                                                const T *beta_data, T *output_data,
                                                size_t batch_size, size_t channels,
                                                size_t spatial_size, T epsilon, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  T mean_val = running_mean_data[c];
  T var_val = running_var_data[c];
  T std_val = sqrt(var_val + epsilon);
  const T inv_std = T(1) / std_val;

  T input_val = input_data[idx];
  T normalized_val = (input_val - mean_val) * inv_std;

  if (affine) {
    const T gamma_val = gamma_data[c];
    const T beta_val = beta_data[c];
    output_data[idx] = gamma_val * normalized_val + beta_val;
  } else {
    output_data[idx] = normalized_val;
  }
}

template <typename T>
__global__ void compute_grad_normalized_kernel(const T *gradient_data, const T *gamma_data,
                                               T *grad_normalized_data, size_t batch_size,
                                               size_t channels, size_t spatial_size, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  if (affine) {
    int c = (idx / spatial_size) % channels;
    const T gamma_val = gamma_data[c];
    grad_normalized_data[idx] = gradient_data[idx] * gamma_val;
  } else {
    grad_normalized_data[idx] = gradient_data[idx];
  }
}

template <typename T>
__global__ void compute_backward_sums_kernel(const T *grad_normalized_data,
                                             const T *normalized_data, T *sum_grad_normalized_data,
                                             T *sum_grad_norm_times_norm_data, size_t batch_size,
                                             size_t channels, size_t spatial_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;

  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T sum_grad_norm = T(0);
  T sum_grad_norm_x_norm = T(0);

  for (size_t n = 0; n < batch_size; ++n) {
    const size_t base_idx = n * channel_stride + c_offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      size_t idx = base_idx + i;
      sum_grad_norm += grad_normalized_data[idx];
      sum_grad_norm_x_norm += grad_normalized_data[idx] * normalized_data[idx];
    }
  }

  sum_grad_normalized_data[c] = sum_grad_norm;
  sum_grad_norm_times_norm_data[c] = sum_grad_norm_x_norm;
}

template <typename T>
__global__ void compute_input_gradients_batchnorm_kernel(
    const T *grad_normalized_data, const T *normalized_data, const T *std_data,
    const T *sum_grad_normalized_data, const T *sum_grad_norm_times_norm_data, T *grad_input_data,
    size_t batch_size, size_t channels, size_t spatial_size, size_t total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * channels * spatial_size;

  if (idx >= total_size)
    return;

  int c = (idx / spatial_size) % channels;

  const T std_val_c = std_data[c];
  const T inv_std = T(1) / std_val_c;
  const T sum_grad_norm = sum_grad_normalized_data[c];
  const T sum_grad_norm_x_norm = sum_grad_norm_times_norm_data[c];
  const T inv_total = T(1) / static_cast<T>(total_elements);

  // ∂L/∂x = (1/N) * (1/σ) * [N * ∂L/∂x̂ - Σ(∂L/∂x̂) - x̂ * Σ(∂L/∂x̂ * x̂)]
  grad_input_data[idx] = inv_std * inv_total *
                         (static_cast<T>(total_elements) * grad_normalized_data[idx] -
                          sum_grad_norm - normalized_data[idx] * sum_grad_norm_x_norm);
}

// Host functions

template <typename T>
void compute_batch_std(const T *batch_var_data, T *batch_std_data, size_t channels, T epsilon) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_batch_std_kernel<<<num_blocks, threads_per_block>>>(batch_var_data, batch_std_data,
                                                              channels, epsilon);
  cudaDeviceSynchronize();
}

template <typename T>
void update_running_stats(T *running_mean_data, T *running_var_data, const T *batch_mean_data,
                          const T *batch_var_data, size_t channels, T momentum) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  update_running_stats_kernel<<<num_blocks, threads_per_block>>>(
      running_mean_data, running_var_data, batch_mean_data, batch_var_data, channels, momentum);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_inference_output(const T *input_data, const T *running_mean_data,
                              const T *running_var_data, const T *gamma_data, const T *beta_data,
                              T *output_data, size_t batch_size, size_t channels,
                              size_t spatial_size, T epsilon, bool affine) {
  int total_elements = batch_size * channels * spatial_size;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_inference_output_kernel<<<num_blocks, threads_per_block>>>(
      input_data, running_mean_data, running_var_data, gamma_data, beta_data, output_data,
      batch_size, channels, spatial_size, epsilon, affine);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_grad_normalized(const T *gradient_data, const T *gamma_data, T *grad_normalized_data,
                             size_t batch_size, size_t channels, size_t spatial_size, bool affine) {
  int total_elements = batch_size * channels * spatial_size;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_grad_normalized_kernel<<<num_blocks, threads_per_block>>>(
      gradient_data, gamma_data, grad_normalized_data, batch_size, channels, spatial_size, affine);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_backward_sums(const T *grad_normalized_data, const T *normalized_data,
                           T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data,
                           size_t batch_size, size_t channels, size_t spatial_size) {
  int threads_per_block = 256;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_backward_sums_kernel<<<num_blocks, threads_per_block>>>(
      grad_normalized_data, normalized_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, batch_size, channels, spatial_size);
  cudaDeviceSynchronize();
}

template <typename T>
void compute_input_gradients_batchnorm(const T *grad_normalized_data, const T *normalized_data,
                                       const T *std_data, const T *sum_grad_normalized_data,
                                       const T *sum_grad_norm_times_norm_data, T *grad_input_data,
                                       size_t batch_size, size_t channels, size_t spatial_size,
                                       size_t total_elements) {
  int total_size = batch_size * channels * spatial_size;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_kernel<<<num_blocks, threads_per_block>>>(
      grad_normalized_data, normalized_data, std_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements);
  cudaDeviceSynchronize();
}

// Template instantiations for new functions

template void compute_batch_std<float>(const float *batch_var_data, float *batch_std_data,
                                       size_t channels, float epsilon);
template void compute_batch_std<double>(const double *batch_var_data, double *batch_std_data,
                                        size_t channels, double epsilon);

template void update_running_stats<float>(float *running_mean_data, float *running_var_data,
                                          const float *batch_mean_data, const float *batch_var_data,
                                          size_t channels, float momentum);
template void update_running_stats<double>(double *running_mean_data, double *running_var_data,
                                           const double *batch_mean_data,
                                           const double *batch_var_data, size_t channels,
                                           double momentum);

template void
compute_inference_output<float>(const float *input_data, const float *running_mean_data,
                                const float *running_var_data, const float *gamma_data,
                                const float *beta_data, float *output_data, size_t batch_size,
                                size_t channels, size_t spatial_size, float epsilon, bool affine);
template void
compute_inference_output<double>(const double *input_data, const double *running_mean_data,
                                 const double *running_var_data, const double *gamma_data,
                                 const double *beta_data, double *output_data, size_t batch_size,
                                 size_t channels, size_t spatial_size, double epsilon, bool affine);

template void compute_grad_normalized<float>(const float *gradient_data, const float *gamma_data,
                                             float *grad_normalized_data, size_t batch_size,
                                             size_t channels, size_t spatial_size, bool affine);
template void compute_grad_normalized<double>(const double *gradient_data, const double *gamma_data,
                                              double *grad_normalized_data, size_t batch_size,
                                              size_t channels, size_t spatial_size, bool affine);

template void compute_backward_sums<float>(const float *grad_normalized_data,
                                           const float *normalized_data,
                                           float *sum_grad_normalized_data,
                                           float *sum_grad_norm_times_norm_data, size_t batch_size,
                                           size_t channels, size_t spatial_size);
template void compute_backward_sums<double>(const double *grad_normalized_data,
                                            const double *normalized_data,
                                            double *sum_grad_normalized_data,
                                            double *sum_grad_norm_times_norm_data,
                                            size_t batch_size, size_t channels,
                                            size_t spatial_size);

template void compute_input_gradients_batchnorm<float>(
    const float *grad_normalized_data, const float *normalized_data, const float *std_data,
    const float *sum_grad_normalized_data, const float *sum_grad_norm_times_norm_data,
    float *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements);
template void compute_input_gradients_batchnorm<double>(
    const double *grad_normalized_data, const double *normalized_data, const double *std_data,
    const double *sum_grad_normalized_data, const double *sum_grad_norm_times_norm_data,
    double *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements);

} // namespace batchnorm
} // namespace cuda
} // namespace tnn