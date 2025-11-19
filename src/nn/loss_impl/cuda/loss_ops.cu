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

// Helper function to get cuBLAS handle
static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

// CrossEntropy Loss Kernels
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

// Softmax CrossEntropy Loss Kernels
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
  int batch_idx = blockIdx.x;
  int class_idx = threadIdx.x;

  if (batch_idx >= batch_size || class_idx >= num_classes)
    return;

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

// MSE Loss Kernels
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

// MAE Loss Kernels
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

// Huber Loss Kernels
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

// Sum reduction kernel for loss aggregation
template <typename T>
__global__ void sum_reduce_kernel(const T *values, T *result, const size_t size) {
  extern __shared__ char shared_mem[];
  T *shared_data = (T *)shared_mem;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  shared_data[threadIdx.x] = (idx < size) ? values[idx] : T(0);
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, shared_data[0]);
  }
}

// Implementation functions
template <typename T>
T compute_crossentropy_loss(const T *predictions, const T *targets, const size_t batch_size,
                            const size_t num_classes, T epsilon) {
  T *d_loss_values;
  cudaMalloc(&d_loss_values, batch_size * sizeof(T));

  int threads_per_block = 256;
  int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  crossentropy_loss_kernel<T><<<num_blocks, threads_per_block>>>(
      predictions, targets, d_loss_values, batch_size, num_classes, epsilon);

  T *d_total_loss;
  cudaMalloc(&d_total_loss, sizeof(T));
  T h_total_loss = T(0);
  cudaMemcpy(d_total_loss, &h_total_loss, sizeof(T), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;
  sum_reduce_kernel<T>
      <<<grid_size, block_size, block_size * sizeof(T)>>>(d_loss_values, d_total_loss, batch_size);

  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_loss_values);
  cudaFree(d_total_loss);

  return h_total_loss / batch_size;
}

template <typename T>
void compute_crossentropy_gradient(const T *predictions, const T *targets, T *gradient,
                                   const size_t batch_size, const size_t num_classes) {
  T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  size_t total_size = batch_size * num_classes;
  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  crossentropy_gradient_kernel<T><<<num_blocks, threads_per_block>>>(
      predictions, targets, gradient, batch_size, num_classes, inv_batch_size);
}

template <typename T>
T compute_softmax_crossentropy_loss(const T *logits, const T *targets, const size_t batch_size,
                                    const size_t num_classes) {
  T *d_loss_values;
  cudaMalloc(&d_loss_values, batch_size * sizeof(T));

  int threads_per_block = 256;
  int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  softmax_crossentropy_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(logits, targets, d_loss_values, batch_size, num_classes);

  T *d_total_loss;
  cudaMalloc(&d_total_loss, sizeof(T));
  T h_total_loss = T(0);
  cudaMemcpy(d_total_loss, &h_total_loss, sizeof(T), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;
  sum_reduce_kernel<T>
      <<<grid_size, block_size, block_size * sizeof(T)>>>(d_loss_values, d_total_loss, batch_size);

  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_loss_values);
  cudaFree(d_total_loss);

  return h_total_loss / batch_size;
}

template <typename T>
void compute_softmax_crossentropy_gradient(const T *logits, const T *targets, T *gradient,
                                           const size_t batch_size, const size_t num_classes) {
  T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  int threads_per_block = 256;
  int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  if (num_classes <= 256) {
    softmax_crossentropy_gradient_kernel<T><<<num_blocks, num_classes>>>(
        logits, targets, gradient, batch_size, num_classes, inv_batch_size);
  } else {
    size_t total_size = batch_size * num_classes;
    int grid_size = (total_size + threads_per_block - 1) / threads_per_block;
    softmax_crossentropy_gradient_kernel<T><<<grid_size, threads_per_block>>>(
        logits, targets, gradient, batch_size, num_classes, inv_batch_size);
  }
}

template <typename T>
T compute_mse_loss(const T *predictions, const T *targets, const size_t batch_size,
                   const size_t output_size) {
  size_t total_size = batch_size * output_size;
  T *d_loss_values;
  cudaMalloc(&d_loss_values, total_size * sizeof(T));

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mse_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size);

  T *d_total_loss;
  cudaMalloc(&d_total_loss, sizeof(T));
  T h_total_loss = T(0);
  cudaMemcpy(d_total_loss, &h_total_loss, sizeof(T), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (total_size + block_size - 1) / block_size;
  sum_reduce_kernel<T>
      <<<grid_size, block_size, block_size * sizeof(T)>>>(d_loss_values, d_total_loss, total_size);

  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_loss_values);
  cudaFree(d_total_loss);

  return h_total_loss / total_size;
}

template <typename T>
void compute_mse_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size) {
  T scale = static_cast<T>(2.0) / static_cast<T>(batch_size * output_size);
  size_t total_size = batch_size * output_size;

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mse_gradient_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, gradient, total_size, scale);
}

template <typename T>
T compute_mae_loss(const T *predictions, const T *targets, const size_t batch_size,
                   const size_t output_size) {
  size_t total_size = batch_size * output_size;
  T *d_loss_values;
  cudaMalloc(&d_loss_values, total_size * sizeof(T));

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mae_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size);

  T *d_total_loss;
  cudaMalloc(&d_total_loss, sizeof(T));
  T h_total_loss = T(0);
  cudaMemcpy(d_total_loss, &h_total_loss, sizeof(T), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (total_size + block_size - 1) / block_size;
  sum_reduce_kernel<T>
      <<<grid_size, block_size, block_size * sizeof(T)>>>(d_loss_values, d_total_loss, total_size);

  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_loss_values);
  cudaFree(d_total_loss);

  return h_total_loss / total_size;
}

template <typename T>
void compute_mae_gradient(const T *predictions, const T *targets, T *gradient,
                          const size_t batch_size, const size_t output_size) {
  T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
  size_t total_size = batch_size * output_size;

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mae_gradient_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, gradient, total_size, scale);
}

template <typename T>
T compute_huber_loss(const T *predictions, const T *targets, const size_t batch_size,
                     const size_t output_size, T delta) {
  size_t total_size = batch_size * output_size;
  T *d_loss_values;
  cudaMalloc(&d_loss_values, total_size * sizeof(T));

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  huber_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size, delta);

  T *d_total_loss;
  cudaMalloc(&d_total_loss, sizeof(T));
  T h_total_loss = T(0);
  cudaMemcpy(d_total_loss, &h_total_loss, sizeof(T), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (total_size + block_size - 1) / block_size;
  sum_reduce_kernel<T>
      <<<grid_size, block_size, block_size * sizeof(T)>>>(d_loss_values, d_total_loss, total_size);

  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_loss_values);
  cudaFree(d_total_loss);

  return h_total_loss / total_size;
}

template <typename T>
void compute_huber_gradient(const T *predictions, const T *targets, T *gradient,
                            const size_t batch_size, const size_t output_size, T delta) {
  T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
  size_t total_size = batch_size * output_size;

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  huber_gradient_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, gradient, total_size, delta, scale);
}

// Explicit template instantiations
template float compute_crossentropy_loss<float>(const float *predictions, const float *targets,
                                                const size_t batch_size, const size_t num_classes,
                                                float epsilon);
template double compute_crossentropy_loss<double>(const double *predictions, const double *targets,
                                                  const size_t batch_size, const size_t num_classes,
                                                  double epsilon);
template void compute_crossentropy_gradient<float>(const float *predictions, const float *targets,
                                                   float *gradient, const size_t batch_size,
                                                   const size_t num_classes);
template void compute_crossentropy_gradient<double>(const double *predictions,
                                                    const double *targets, double *gradient,
                                                    const size_t batch_size,
                                                    const size_t num_classes);

template float compute_softmax_crossentropy_loss<float>(const float *logits, const float *targets,
                                                        const size_t batch_size,
                                                        const size_t num_classes);
template double compute_softmax_crossentropy_loss<double>(const double *logits,
                                                          const double *targets,
                                                          const size_t batch_size,
                                                          const size_t num_classes);
template void compute_softmax_crossentropy_gradient<float>(const float *logits,
                                                           const float *targets, float *gradient,
                                                           const size_t batch_size,
                                                           const size_t num_classes);
template void compute_softmax_crossentropy_gradient<double>(const double *logits,
                                                            const double *targets, double *gradient,
                                                            const size_t batch_size,
                                                            const size_t num_classes);

template float compute_mse_loss<float>(const float *predictions, const float *targets,
                                       const size_t batch_size, const size_t output_size);
template double compute_mse_loss<double>(const double *predictions, const double *targets,
                                         const size_t batch_size, const size_t output_size);
template void compute_mse_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size);
template void compute_mse_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size);

template float compute_mae_loss<float>(const float *predictions, const float *targets,
                                       const size_t batch_size, const size_t output_size);
template double compute_mae_loss<double>(const double *predictions, const double *targets,
                                         const size_t batch_size, const size_t output_size);
template void compute_mae_gradient<float>(const float *predictions, const float *targets,
                                          float *gradient, const size_t batch_size,
                                          const size_t output_size);
template void compute_mae_gradient<double>(const double *predictions, const double *targets,
                                           double *gradient, const size_t batch_size,
                                           const size_t output_size);

template float compute_huber_loss<float>(const float *predictions, const float *targets,
                                         const size_t batch_size, const size_t output_size,
                                         float delta);
template double compute_huber_loss<double>(const double *predictions, const double *targets,
                                           const size_t batch_size, const size_t output_size,
                                           double delta);
template void compute_huber_gradient<float>(const float *predictions, const float *targets,
                                            float *gradient, const size_t batch_size,
                                            const size_t output_size, float delta);
template void compute_huber_gradient<double>(const double *predictions, const double *targets,
                                             double *gradient, const size_t batch_size,
                                             const size_t output_size, double delta);

} // namespace loss
} // namespace cuda
} // namespace tnn
