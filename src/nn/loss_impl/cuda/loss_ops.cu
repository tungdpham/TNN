/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/loss_impl/cuda/loss_ops.hpp"

#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace loss {

template <typename T>
__global__ void crossentropy_loss_kernel(const T *predictions, const T *targets, T *loss_values,
                                         const size_t batch_size, const size_t num_classes,
                                         T epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  T batch_loss = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    if (targets[idx * num_classes + j] > T(0.5)) {
      T pred = predictions[idx * num_classes + j];
      pred = fmax(pred, epsilon);
      pred = fmin(pred, T(1.0) - epsilon);
      batch_loss -= log(pred);
      break;
    }
  }

  loss_values[idx] = batch_loss;
}

template <>
__global__ void crossentropy_loss_kernel<double>(const double *predictions, const double *targets,
                                                 double *loss_values, const size_t batch_size,
                                                 const size_t num_classes, double epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  double batch_loss = 0.0;
  for (size_t j = 0; j < num_classes; ++j) {
    if (targets[idx * num_classes + j] > 0.5) {
      double pred = predictions[idx * num_classes + j];
      pred = fmax(pred, epsilon);
      pred = fmin(pred, 1.0 - epsilon);
      batch_loss -= log(pred);
      break;
    }
  }

  loss_values[idx] = batch_loss;
}

template <typename T>
__global__ void crossentropy_gradient_kernel(const T *predictions, const T *targets, T *gradient,
                                             const size_t batch_size, const size_t num_classes,
                                             T inv_batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * num_classes)
    return;

  gradient[idx] = (predictions[idx] - targets[idx]) * inv_batch_size;
}

template <typename T>
__global__ void softmax_crossentropy_loss_kernel(const T *logits, const T *targets, T *loss_values,
                                                 const size_t batch_size,
                                                 const size_t num_classes) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size)
    return;

  T max_logit = logits[batch_idx * num_classes];
  for (size_t j = 1; j < num_classes; ++j) {
    max_logit = fmax(max_logit, logits[batch_idx * num_classes + j]);
  }

  T sum_exp = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    sum_exp += exp(logits[batch_idx * num_classes + j] - max_logit);
  }
  T log_sum_exp = log(sum_exp) + max_logit;

  T batch_loss = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    if (targets[batch_idx * num_classes + j] > T(0.5)) {
      batch_loss = log_sum_exp - logits[batch_idx * num_classes + j];
      break;
    }
  }

  loss_values[batch_idx] = batch_loss;
}

template <typename T>
__global__ void softmax_crossentropy_gradient_kernel(const T *logits, const T *targets, T *gradient,
                                                     const size_t batch_size,
                                                     const size_t num_classes, T inv_batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * num_classes)
    return;

  int batch_idx = idx / num_classes;
  int class_idx = idx % num_classes;

  T max_logit = logits[batch_idx * num_classes];
  for (size_t j = 1; j < num_classes; ++j) {
    max_logit = fmax(max_logit, logits[batch_idx * num_classes + j]);
  }

  T sum_exp = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    sum_exp += exp(logits[batch_idx * num_classes + j] - max_logit);
  }

  T softmax_prob = exp(logits[batch_idx * num_classes + class_idx] - max_logit) / sum_exp;
  gradient[batch_idx * num_classes + class_idx] =
      (softmax_prob - targets[batch_idx * num_classes + class_idx]) * inv_batch_size;
}

template <typename T>
__global__ void logsoftmax_crossentropy_loss_kernel(const T *logits, const T *targets,
                                                    T *loss_values, const size_t batch_size,
                                                    const size_t num_classes) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size)
    return;

  // Find max logit for numerical stability
  T max_logit = logits[batch_idx * num_classes];
  for (size_t j = 1; j < num_classes; ++j) {
    max_logit = fmax(max_logit, logits[batch_idx * num_classes + j]);
  }

  // Compute log-sum-exp
  T sum_exp = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    sum_exp += exp(logits[batch_idx * num_classes + j] - max_logit);
  }
  T log_sum_exp = log(sum_exp) + max_logit;

  // Compute loss (negative log likelihood)
  T batch_loss = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    if (targets[batch_idx * num_classes + j] > T(0.5)) {
      // log_softmax = logit - log_sum_exp
      // CrossEntropy = -target * log_softmax = -(logit - log_sum_exp)
      batch_loss = log_sum_exp - logits[batch_idx * num_classes + j];
      break;
    }
  }

  loss_values[batch_idx] = batch_loss;
}

template <typename T>
__global__ void logsoftmax_crossentropy_gradient_kernel(const T *logits, const T *targets,
                                                        T *gradient, const size_t batch_size,
                                                        const size_t num_classes,
                                                        T inv_batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * num_classes)
    return;

  int batch_idx = idx / num_classes;
  int class_idx = idx % num_classes;

  // Find max logit for numerical stability
  T max_logit = logits[batch_idx * num_classes];
  for (size_t j = 1; j < num_classes; ++j) {
    max_logit = fmax(max_logit, logits[batch_idx * num_classes + j]);
  }

  // Compute sum_exp
  T sum_exp = T(0);
  for (size_t j = 0; j < num_classes; ++j) {
    sum_exp += exp(logits[batch_idx * num_classes + j] - max_logit);
  }

  // Gradient: softmax(logit) - target
  T softmax_prob = exp(logits[batch_idx * num_classes + class_idx] - max_logit) / sum_exp;
  gradient[batch_idx * num_classes + class_idx] =
      (softmax_prob - targets[batch_idx * num_classes + class_idx]) * inv_batch_size;
}

template <typename T>
__global__ void mse_loss_kernel(const T *predictions, const T *targets, T *loss_values,
                                const size_t total_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  T diff = predictions[idx] - targets[idx];
  loss_values[idx] = diff * diff;
}

template <typename T>
__global__ void mse_gradient_kernel(const T *predictions, const T *targets, T *gradient,
                                    const size_t total_size, T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  gradient[idx] = (predictions[idx] - targets[idx]) * scale;
}

template <typename T>
__global__ void mae_loss_kernel(const T *predictions, const T *targets, T *loss_values,
                                const size_t total_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  loss_values[idx] = abs(predictions[idx] - targets[idx]);
}

template <typename T>
__global__ void mae_gradient_kernel(const T *predictions, const T *targets, T *gradient,
                                    const size_t total_size, T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  T diff = predictions[idx] - targets[idx];
  gradient[idx] = (diff > T(0) ? scale : -scale);
}

template <typename T>
__global__ void huber_loss_kernel(const T *predictions, const T *targets, T *loss_values,
                                  const size_t total_size, T delta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  T diff = abs(predictions[idx] - targets[idx]);
  if (diff <= delta) {
    loss_values[idx] = T(0.5) * diff * diff;
  } else {
    loss_values[idx] = delta * diff - T(0.5) * delta * delta;
  }
}

template <typename T>
__global__ void huber_gradient_kernel(const T *predictions, const T *targets, T *gradient,
                                      const size_t total_size, T delta, T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  T diff = predictions[idx] - targets[idx];
  T abs_diff = abs(diff);

  if (abs_diff <= delta) {
    gradient[idx] = diff * scale;
  } else {
    gradient[idx] = (diff > T(0) ? delta : -delta) * scale;
  }
}

template <typename T>
__global__ void sum_reduce_kernel_stage1(const T *values, T *block_results, const size_t size) {
  extern __shared__ char shared_mem[];
  T *shared_data = (T *)shared_mem;

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  T local_sum = T(0);
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    local_sum += values[i];
  }

  shared_data[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_results[blockIdx.x] = shared_data[0];
  }
}

template <typename T>
__global__ void sum_reduce_kernel_stage2(const T *block_results, T *result,
                                         const size_t num_blocks) {
  extern __shared__ char shared_mem[];
  T *shared_data = (T *)shared_mem;

  int tid = threadIdx.x;

  T local_sum = T(0);
  for (int i = tid; i < num_blocks; i += blockDim.x) {
    local_sum += block_results[i];
  }

  shared_data[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = shared_data[0];
  }
}

template <typename T>
void compute_crossentropy_loss(const T *predictions, const T *targets, T &loss,
                               const size_t batch_size, const size_t num_classes, T epsilon,
                               cudaStream_t stream) {
  T *d_loss_values;
  cudaMallocAsync(&d_loss_values, batch_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  crossentropy_loss_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      predictions, targets, d_loss_values, batch_size, num_classes, epsilon);

  int block_size = 256;
  int grid_size = std::min(256, (int)((batch_size + block_size - 1) / block_size));

  T *d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(T), stream);

  sum_reduce_kernel_stage1<T><<<grid_size, block_size, block_size * sizeof(T), stream>>>(
      d_loss_values, d_block_results, batch_size);

  T *d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(T), stream);

  sum_reduce_kernel_stage2<T>
      <<<1, block_size, block_size * sizeof(T), stream>>>(d_block_results, d_total_loss, grid_size);

  T h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = h_total_loss / batch_size;
}

template <typename T>
void compute_crossentropy_gradient(const T *predictions, const T *targets, T *gradient,
                                   const size_t batch_size, const size_t num_classes,
                                   cudaStream_t stream) {
  T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  size_t total_size = batch_size * num_classes;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  crossentropy_gradient_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      predictions, targets, gradient, batch_size, num_classes, inv_batch_size);

  cudaStreamSynchronize(stream);
}

template <typename T>
void compute_softmax_crossentropy_loss(const T *logits, const T *targets, T &loss,
                                       const size_t batch_size, const size_t num_classes,
                                       cudaStream_t stream) {
  T *d_loss_values;
  cudaMallocAsync(&d_loss_values, batch_size * sizeof(T), stream);
  cudaMemsetAsync(d_loss_values, 0, batch_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  softmax_crossentropy_loss_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      logits, targets, d_loss_values, batch_size, num_classes);

  int block_size = 256;
  int grid_size = std::min(256, (int)((batch_size + block_size - 1) / block_size));

  T *d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(T), stream);
  cudaMemsetAsync(d_block_results, 0, grid_size * sizeof(T), stream);

  sum_reduce_kernel_stage1<T><<<grid_size, block_size, block_size * sizeof(T), stream>>>(
      d_loss_values, d_block_results, batch_size);

  T *d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(T), stream);
  cudaMemsetAsync(d_total_loss, 0, sizeof(T), stream);

  sum_reduce_kernel_stage2<T>
      <<<1, block_size, block_size * sizeof(T), stream>>>(d_block_results, d_total_loss, grid_size);

  T h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = h_total_loss / batch_size;
}

template <typename T>
void compute_softmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                           const size_t batch_size, const size_t num_classes,
                                           cudaStream_t stream) {
  T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  size_t total_size = batch_size * num_classes;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  softmax_crossentropy_gradient_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      logits, targets, gradient, batch_size, num_classes, inv_batch_size);

  cudaStreamSynchronize(stream);
}

template <typename T>
void compute_logsoftmax_crossentropy_loss(const T *logits, const T *targets, T &loss,
                                          const size_t batch_size, const size_t num_classes,
                                          cudaStream_t stream) {
  T *d_loss_values;
  cudaMallocAsync(&d_loss_values, batch_size * sizeof(T), stream);
  cudaMemsetAsync(d_loss_values, 0, batch_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  logsoftmax_crossentropy_loss_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      logits, targets, d_loss_values, batch_size, num_classes);

  int block_size = 256;
  int grid_size = std::min(256, (int)((batch_size + block_size - 1) / block_size));

  T *d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(T), stream);
  cudaMemsetAsync(d_block_results, 0, grid_size * sizeof(T), stream);
  sum_reduce_kernel_stage1<T><<<grid_size, block_size, block_size * sizeof(T), stream>>>(
      d_loss_values, d_block_results, batch_size);

  T *d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(T), stream);
  cudaMemsetAsync(d_total_loss, 0, sizeof(T), stream);

  sum_reduce_kernel_stage2<T>
      <<<1, block_size, block_size * sizeof(T), stream>>>(d_block_results, d_total_loss, grid_size);

  T h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = h_total_loss / batch_size;
}

template <typename T>
void compute_logsoftmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                              const size_t batch_size, const size_t num_classes,
                                              cudaStream_t stream) {
  T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  size_t total_size = batch_size * num_classes;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  logsoftmax_crossentropy_gradient_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      logits, targets, gradient, batch_size, num_classes, inv_batch_size);

  cudaStreamSynchronize(stream);
}

template <typename T>
void compute_mse_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream) {
  size_t total_size = batch_size * output_size;
  T *d_loss_values;
  cudaMallocAsync(&d_loss_values, total_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mse_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_size + block_size - 1) / block_size));

  T *d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(T), stream);

  sum_reduce_kernel_stage1<T><<<grid_size, block_size, block_size * sizeof(T), stream>>>(
      d_loss_values, d_block_results, total_size);

  T *d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(T), stream);

  sum_reduce_kernel_stage2<T>
      <<<1, block_size, block_size * sizeof(T), stream>>>(d_block_results, d_total_loss, grid_size);

  T h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost, stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = h_total_loss / total_size;
}

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size, cudaStream_t stream) {
  T scale = static_cast<T>(2.0) / static_cast<T>(batch_size * output_size);
  size_t total_size = batch_size * output_size;

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mse_gradient_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, gradient, total_size, scale);

  cudaDeviceSynchronize();
}

template <typename T>
void compute_mae_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream) {
  size_t total_size = batch_size * output_size;
  T *d_loss_values;
  cudaMallocAsync(&d_loss_values, total_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mae_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_size + block_size - 1) / block_size));

  T *d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(T), stream);

  sum_reduce_kernel_stage1<T><<<grid_size, block_size, block_size * sizeof(T), stream>>>(
      d_loss_values, d_block_results, total_size);

  T *d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(T), stream);

  sum_reduce_kernel_stage2<T>
      <<<1, block_size, block_size * sizeof(T), stream>>>(d_block_results, d_total_loss, grid_size);

  T h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost, stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = h_total_loss / total_size;
}

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size, cudaStream_t stream) {
  T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
  size_t total_size = batch_size * output_size;

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mae_gradient_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, gradient, total_size, scale);

  cudaDeviceSynchronize();
}

template <typename T>
void compute_huber_loss(const T *predictions, const T *targets, T &loss, const size_t batch_size,
                        const size_t output_size, T delta, cudaStream_t stream) {
  size_t total_size = batch_size * output_size;
  T *d_loss_values;
  cudaMallocAsync(&d_loss_values, total_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  huber_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size, delta);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_size + block_size - 1) / block_size));

  T *d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(T), stream);

  sum_reduce_kernel_stage1<T><<<grid_size, block_size, block_size * sizeof(T), stream>>>(
      d_loss_values, d_block_results, total_size);

  T *d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(T), stream);

  sum_reduce_kernel_stage2<T>
      <<<1, block_size, block_size * sizeof(T), stream>>>(d_block_results, d_total_loss, grid_size);

  T h_total_loss;
  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = h_total_loss / total_size;
}

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *gradient,
                            const size_t batch_size, const size_t output_size, T delta,
                            cudaStream_t stream) {
  T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
  size_t total_size = batch_size * output_size;

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  huber_gradient_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, gradient, total_size, delta, scale);

  cudaDeviceSynchronize();
}

template void compute_crossentropy_loss<float>(const float *predictions, const float *targets,
                                               float &loss, const size_t batch_size,
                                               const size_t num_classes, float epsilon,
                                               cudaStream_t stream);
template void compute_crossentropy_loss<double>(const double *predictions, const double *targets,
                                                double &loss, const size_t batch_size,
                                                const size_t num_classes, double epsilon,
                                                cudaStream_t stream);
template void compute_crossentropy_gradient<float>(const float *predictions, const float *targets,
                                                   float *gradient, const size_t batch_size,
                                                   const size_t num_classes, cudaStream_t stream);
template void compute_crossentropy_gradient<double>(const double *predictions,
                                                    const double *targets, double *gradient,
                                                    const size_t batch_size,
                                                    const size_t num_classes, cudaStream_t stream);

template void compute_softmax_crossentropy_loss<float>(const float *logits, const float *targets,
                                                       float &loss, const size_t batch_size,
                                                       const size_t num_classes,
                                                       cudaStream_t stream);
template void compute_softmax_crossentropy_loss<double>(const double *logits, const double *targets,
                                                        double &loss, const size_t batch_size,
                                                        const size_t num_classes,
                                                        cudaStream_t stream);
template void compute_softmax_crossentropy_gradient<float>(const float *logits,
                                                           const float *targets, float *gradient,
                                                           const size_t batch_size,
                                                           const size_t num_classes,
                                                           cudaStream_t stream);
template void compute_softmax_crossentropy_gradient<double>(const double *logits,
                                                            const double *targets, double *gradient,
                                                            const size_t batch_size,
                                                            const size_t num_classes,
                                                            cudaStream_t stream);

template void compute_logsoftmax_crossentropy_loss<float>(const float *logits, const float *targets,
                                                          float &loss, const size_t batch_size,
                                                          const size_t num_classes,
                                                          cudaStream_t stream);
template void compute_logsoftmax_crossentropy_loss<double>(const double *logits,
                                                           const double *targets, double &loss,
                                                           const size_t batch_size,
                                                           const size_t num_classes,
                                                           cudaStream_t stream);
template void compute_logsoftmax_crossentropy_gradient<float>(const float *logits,
                                                              const float *targets, float *gradient,
                                                              const size_t batch_size,
                                                              const size_t num_classes,
                                                              cudaStream_t stream);
template void
compute_logsoftmax_crossentropy_gradient<double>(const double *logits, const double *targets,
                                                 double *gradient, const size_t batch_size,
                                                 const size_t num_classes, cudaStream_t stream);

template void compute_mse_loss<float>(const float *predictions, const float *targets, float &loss,
                                      const size_t batch_size, const size_t output_size,
                                      cudaStream_t stream);
template void compute_mse_loss<double>(const double *predictions, const double *targets,
                                       double &loss, const size_t batch_size,
                                       const size_t output_size, cudaStream_t stream);
template void compute_mse_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size, cudaStream_t stream);
template void compute_mse_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size, cudaStream_t stream);

template void compute_mae_loss<float>(const float *predictions, const float *targets, float &loss,
                                      const size_t batch_size, const size_t output_size,
                                      cudaStream_t stream);
template void compute_mae_loss<double>(const double *predictions, const double *targets,
                                       double &loss, const size_t batch_size,
                                       const size_t output_size, cudaStream_t stream);
template void compute_mae_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size, cudaStream_t stream);
template void compute_mae_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size, cudaStream_t stream);

template void compute_huber_loss<float>(const float *predictions, const float *targets, float &loss,
                                        const size_t batch_size, const size_t output_size,
                                        float delta, cudaStream_t stream);
template void compute_huber_loss<double>(const double *predictions, const double *targets,
                                         double &loss, const size_t batch_size,
                                         const size_t output_size, double delta,
                                         cudaStream_t stream);
template void compute_huber_gradient<float>(const float *predictions, const float *targets,
                                            float *gradient, const size_t batch_size,
                                            const size_t output_size, float delta,
                                            cudaStream_t stream);
template void compute_huber_gradient<double>(const double *predictions, const double *targets,
                                             double *gradient, const size_t batch_size,
                                             const size_t output_size, double delta,
                                             cudaStream_t stream);

} // namespace loss
} // namespace cuda
} // namespace tnn
