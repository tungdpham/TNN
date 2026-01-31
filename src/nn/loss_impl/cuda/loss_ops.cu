/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>

#include "nn/loss_impl/cuda/loss_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace loss {

template <typename T>
__global__ void crossentropy_loss_kernel(const T* predictions, const T* targets, T* loss_values,
                                         const size_t total_instances, const size_t num_classes,
                                         T epsilon) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_instances) return;

  size_t b = idx;
  size_t base_ptr = b * num_classes;

  ComputeT batch_loss = ComputeT(0);
  for (size_t c = 0; c < num_classes; ++c) {
    if (targets[base_ptr + c] > T(0)) {
      ComputeT pred = static_cast<ComputeT>(predictions[base_ptr + c]);
      ComputeT eps = static_cast<ComputeT>(epsilon);
      pred = fmaxf(pred, eps);
      pred = fminf(pred, ComputeT(1.0) - eps);
      batch_loss -= static_cast<ComputeT>(targets[base_ptr + c]) * logf(pred);
    }
  }

  loss_values[idx] = static_cast<T>(batch_loss);
}

template <>
__global__ void crossentropy_loss_kernel<double>(const double* predictions, const double* targets,
                                                 double* loss_values, const size_t total_instances,
                                                 const size_t num_classes, double epsilon) {
  using ComputeT = typename TypeTraits<double>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_instances) return;

  size_t b = idx;
  size_t base_ptr = b * num_classes;

  ComputeT batch_loss = 0.0;
  for (size_t c = 0; c < num_classes; ++c) {
    if (targets[base_ptr + c] > 0.0) {
      ComputeT pred = static_cast<ComputeT>(predictions[base_ptr + c]);
      pred = fmax(pred, static_cast<ComputeT>(epsilon));
      pred = fmin(pred, 1.0 - static_cast<ComputeT>(epsilon));
      batch_loss -= static_cast<ComputeT>(targets[base_ptr + c]) * log(pred);
    }
  }

  loss_values[idx] = static_cast<double>(batch_loss);
}

template <typename T>
__global__ void crossentropy_gradient_kernel(const T* predictions, const T* targets, T* gradient,
                                             const size_t total_elements, T epsilon,
                                             T inv_batch_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  ComputeT pred = static_cast<ComputeT>(predictions[idx]);
  ComputeT eps = static_cast<ComputeT>(epsilon);
  pred = fmaxf(pred, eps);
  pred = fminf(pred, ComputeT(1.0) - eps);

  gradient[idx] = static_cast<T>((-static_cast<ComputeT>(targets[idx]) / pred) *
                                 static_cast<ComputeT>(inv_batch_size));
}

template <typename T>
__global__ void logsoftmax_crossentropy_loss_kernel(const T* logits, const T* targets,
                                                    T* loss_values, const size_t total_instances,
                                                    const size_t num_classes) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_instances) return;

  size_t b = idx;
  size_t base_ptr = b * num_classes;

  ComputeT max_logit = static_cast<ComputeT>(logits[base_ptr + 0]);
  for (size_t c = 1; c < num_classes; ++c) {
    max_logit = fmaxf(max_logit, static_cast<ComputeT>(logits[base_ptr + c]));
  }

  ComputeT sum_exp = ComputeT(0);
  for (size_t c = 0; c < num_classes; ++c) {
    sum_exp += expf(static_cast<ComputeT>(logits[base_ptr + c]) - max_logit);
  }
  ComputeT log_sum_exp = logf(sum_exp) + max_logit;

  ComputeT instance_loss = ComputeT(0);
  for (size_t c = 0; c < num_classes; ++c) {
    if (targets[base_ptr + c] > T(0)) {
      instance_loss += static_cast<ComputeT>(targets[base_ptr + c]) *
                       (log_sum_exp - static_cast<ComputeT>(logits[base_ptr + c]));
    }
  }

  loss_values[idx] = static_cast<T>(instance_loss);
}

template <typename T>
__global__ void batch_max_kernel(const T* logits,
                                 typename TypeTraits<T>::ComputePrecision* max_vals,
                                 const size_t num_classes) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t idx = blockIdx.x;
  size_t b = idx;
  size_t base_ptr = b * num_classes;

  ComputeT thread_max = ComputeT(-1e37);
  if (threadIdx.x < num_classes) {
    thread_max = static_cast<ComputeT>(logits[base_ptr + threadIdx.x]);
  }

  for (size_t c = threadIdx.x + blockDim.x; c < num_classes; c += blockDim.x) {
    ComputeT val = static_cast<ComputeT>(logits[base_ptr + c]);
    thread_max = (thread_max > val) ? thread_max : val;
  }

  __shared__ ComputeT shared_max[256];
  shared_max[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ComputeT val = shared_max[threadIdx.x + stride];
      shared_max[threadIdx.x] = (shared_max[threadIdx.x] > val) ? shared_max[threadIdx.x] : val;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    max_vals[idx] = shared_max[0];
  }
}

template <typename T>
__global__ void batch_sumexp_kernel(const T* logits,
                                    const typename TypeTraits<T>::ComputePrecision* max_vals,
                                    typename TypeTraits<T>::ComputePrecision* sum_vals,
                                    const size_t num_classes) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t idx = blockIdx.x;
  size_t b = idx;
  size_t base_ptr = b * num_classes;

  ComputeT instance_max = max_vals[idx];
  ComputeT thread_sum = ComputeT(0);

  for (size_t c = threadIdx.x; c < num_classes; c += blockDim.x) {
    ComputeT val = static_cast<ComputeT>(logits[base_ptr + c]);
    thread_sum += expf(val - instance_max);
  }

  __shared__ ComputeT shared_sum[256];
  shared_sum[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    sum_vals[idx] = shared_sum[0];
  }
}

template <typename T>
__global__ void logsoftmax_crossentropy_gradient_kernel(
    const T* logits, const T* targets, T* gradient,
    const typename TypeTraits<T>::ComputePrecision* max_vals,
    const typename TypeTraits<T>::ComputePrecision* sum_vals, const size_t total_elements,
    const size_t num_classes, typename TypeTraits<T>::ComputePrecision inv_total_instances) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t b = idx / num_classes;

  ComputeT max_logit = max_vals[b];
  ComputeT sum_exp = sum_vals[b];

  ComputeT softmax_prob = expf(static_cast<ComputeT>(logits[idx]) - max_logit) / sum_exp;
  gradient[idx] =
      static_cast<T>((softmax_prob - static_cast<ComputeT>(targets[idx])) * inv_total_instances);
}

template <typename T>
__global__ void mse_loss_kernel(const T* predictions, const T* targets, T* loss_values,
                                const size_t total_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  ComputeT diff = static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
  loss_values[idx] = static_cast<T>(diff * diff);
}

template <typename T>
__global__ void mse_gradient_kernel(const T* predictions, const T* targets, T* gradient,
                                    const size_t total_size, T scale) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  gradient[idx] = static_cast<T>(
      (static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx])) *
      static_cast<ComputeT>(scale));
}

template <typename T>
__global__ void mae_loss_kernel(const T* predictions, const T* targets, T* loss_values,
                                const size_t total_size) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  ComputeT diff = static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
  loss_values[idx] = static_cast<T>((diff > ComputeT(0)) ? diff : -diff);
}

template <typename T>
__global__ void mae_gradient_kernel(const T* predictions, const T* targets, T* gradient,
                                    const size_t total_size, T scale) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  ComputeT diff = static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
  gradient[idx] = static_cast<T>(
      (diff > ComputeT(0) ? static_cast<ComputeT>(scale) : -static_cast<ComputeT>(scale)));
}

template <typename T>
__global__ void huber_loss_kernel(const T* predictions, const T* targets, T* loss_values,
                                  const size_t total_size, T delta) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  ComputeT diff = static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
  ComputeT abs_diff = (diff > ComputeT(0)) ? diff : -diff;
  ComputeT delta_c = static_cast<ComputeT>(delta);
  if (abs_diff <= delta_c) {
    loss_values[idx] = static_cast<T>(ComputeT(0.5) * abs_diff * abs_diff);
  } else {
    loss_values[idx] = static_cast<T>(delta_c * abs_diff - ComputeT(0.5) * delta_c * delta_c);
  }
}

template <typename T>
__global__ void huber_gradient_kernel(const T* predictions, const T* targets, T* gradient,
                                      const size_t total_size, T delta, T scale) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  ComputeT diff = static_cast<ComputeT>(predictions[idx]) - static_cast<ComputeT>(targets[idx]);
  ComputeT abs_diff = (diff > ComputeT(0)) ? diff : -diff;
  ComputeT delta_c = static_cast<ComputeT>(delta);
  ComputeT scale_c = static_cast<ComputeT>(scale);

  if (abs_diff <= delta_c) {
    gradient[idx] = static_cast<T>(diff * scale_c);
  } else {
    gradient[idx] = static_cast<T>((diff > ComputeT(0) ? delta_c : -delta_c) * scale_c);
  }
}

template <typename T>
__global__ void sum_reduce_kernel_stage1(const T* values, T* block_results, const size_t size) {
  extern __shared__ char shared_mem[];
  T* shared_data = (T*)shared_mem;

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

// Overload for mixed types: reduce from InputT to OutputT (for higher precision accumulation)
template <typename OutputT, typename InputT>
__global__ void sum_reduce_kernel_stage1(const InputT* values, OutputT* block_results,
                                         const size_t size) {
  extern __shared__ char shared_mem[];
  OutputT* shared_data = (OutputT*)shared_mem;

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  OutputT local_sum = OutputT(0);
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    local_sum += static_cast<OutputT>(values[i]);
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
__global__ void sum_reduce_kernel_stage2(const T* block_results, T* result,
                                         const size_t num_blocks) {
  extern __shared__ char shared_mem[];
  T* shared_data = (T*)shared_mem;

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
void compute_crossentropy_loss(const T* predictions, const T* targets, float& loss,
                               const size_t batch_size, const size_t num_classes, T epsilon,
                               cudaStream_t stream) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t total_instances = batch_size;
  T* d_loss_values;
  cudaMallocAsync(&d_loss_values, total_instances * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_instances + threads_per_block - 1) / threads_per_block;

  crossentropy_loss_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      predictions, targets, d_loss_values, total_instances, num_classes, epsilon);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_instances + block_size - 1) / block_size));

  ComputeT* d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(ComputeT), stream);

  sum_reduce_kernel_stage1<ComputeT>
      <<<grid_size, block_size, block_size * sizeof(ComputeT), stream>>>(
          d_loss_values, d_block_results, total_instances);

  ComputeT* d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(ComputeT), stream);

  sum_reduce_kernel_stage2<ComputeT><<<1, block_size, block_size * sizeof(ComputeT), stream>>>(
      d_block_results, d_total_loss, grid_size);

  ComputeT h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(ComputeT), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = static_cast<float>(h_total_loss) / static_cast<float>(total_instances);
}

template <typename T>
void compute_crossentropy_gradient(const T* predictions, const T* targets, T* gradient,
                                   const size_t batch_size, const size_t num_classes, T epsilon,
                                   cudaStream_t stream) {
  size_t total_elements = batch_size * num_classes;
  T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  crossentropy_gradient_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      predictions, targets, gradient, total_elements, epsilon, inv_batch_size);

  cudaStreamSynchronize(stream);
}

template <typename T>
void compute_logsoftmax_crossentropy_loss(const T* logits, const T* targets, float& loss,
                                          const size_t batch_size, const size_t num_classes,
                                          cudaStream_t stream) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t total_instances = batch_size;
  T* d_loss_values;
  cudaMallocAsync(&d_loss_values, total_instances * sizeof(T), stream);
  cudaMemsetAsync(d_loss_values, 0, total_instances * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_instances + threads_per_block - 1) / threads_per_block;

  logsoftmax_crossentropy_loss_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      logits, targets, d_loss_values, total_instances, num_classes);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_instances + block_size - 1) / block_size));

  ComputeT* d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(ComputeT), stream);
  cudaMemsetAsync(d_block_results, 0, grid_size * sizeof(ComputeT), stream);
  sum_reduce_kernel_stage1<ComputeT>
      <<<grid_size, block_size, block_size * sizeof(ComputeT), stream>>>(
          d_loss_values, d_block_results, total_instances);

  ComputeT* d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(ComputeT), stream);
  cudaMemsetAsync(d_total_loss, 0, sizeof(ComputeT), stream);

  sum_reduce_kernel_stage2<ComputeT><<<1, block_size, block_size * sizeof(ComputeT), stream>>>(
      d_block_results, d_total_loss, grid_size);

  ComputeT h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(ComputeT), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = static_cast<float>(h_total_loss) / static_cast<float>(total_instances);
}

template <typename T>
void compute_logsoftmax_crossentropy_gradient(const T* logits, const T* targets, T* gradient,
                                              const size_t batch_size, const size_t num_classes,
                                              cudaStream_t stream) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t total_instances = batch_size;
  ComputeT inv_total_instances =
      static_cast<ComputeT>(1.0) / static_cast<ComputeT>(total_instances);

  ComputeT* d_max_vals;
  ComputeT* d_sum_vals;
  cudaMallocAsync(&d_max_vals, total_instances * sizeof(ComputeT), stream);
  cudaMallocAsync(&d_sum_vals, total_instances * sizeof(ComputeT), stream);

  batch_max_kernel<T><<<total_instances, 256, 0, stream>>>(logits, d_max_vals, num_classes);

  batch_sumexp_kernel<T>
      <<<total_instances, 256, 0, stream>>>(logits, d_max_vals, d_sum_vals, num_classes);

  size_t total_elements = batch_size * num_classes;

  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  logsoftmax_crossentropy_gradient_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      logits, targets, gradient, d_max_vals, d_sum_vals, total_elements, num_classes,
      inv_total_instances);

  cudaFreeAsync(d_max_vals, stream);
  cudaFreeAsync(d_sum_vals, stream);

  cudaStreamSynchronize(stream);
}

template <typename T>
void compute_mse_loss(const T* predictions, const T* targets, float& loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t total_size = batch_size * output_size;
  T* d_loss_values;
  cudaMallocAsync(&d_loss_values, total_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mse_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_size + block_size - 1) / block_size));

  ComputeT* d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(ComputeT), stream);

  sum_reduce_kernel_stage1<ComputeT>
      <<<grid_size, block_size, block_size * sizeof(ComputeT), stream>>>(
          d_loss_values, d_block_results, total_size);

  ComputeT* d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(ComputeT), stream);

  sum_reduce_kernel_stage2<ComputeT><<<1, block_size, block_size * sizeof(ComputeT), stream>>>(
      d_block_results, d_total_loss, grid_size);

  ComputeT h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(ComputeT), cudaMemcpyDeviceToHost, stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = static_cast<float>(h_total_loss) / static_cast<float>(total_size);
}

template <typename T>
void compute_mse_gradient(const T* predictions, const T* targets, T* gradient,
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
void compute_mae_loss(const T* predictions, const T* targets, float& loss, const size_t batch_size,
                      const size_t output_size, cudaStream_t stream) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t total_size = batch_size * output_size;
  T* d_loss_values;
  cudaMallocAsync(&d_loss_values, total_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  mae_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_size + block_size - 1) / block_size));

  ComputeT* d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(ComputeT), stream);

  sum_reduce_kernel_stage1<ComputeT>
      <<<grid_size, block_size, block_size * sizeof(ComputeT), stream>>>(
          d_loss_values, d_block_results, total_size);

  ComputeT* d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(ComputeT), stream);

  sum_reduce_kernel_stage2<ComputeT><<<1, block_size, block_size * sizeof(ComputeT), stream>>>(
      d_block_results, d_total_loss, grid_size);

  ComputeT h_total_loss;
  cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(ComputeT), cudaMemcpyDeviceToHost, stream);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = static_cast<float>(h_total_loss) / static_cast<float>(total_size);
}

template <typename T>
void compute_mae_gradient(const T* predictions, const T* targets, T* gradient,
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
void compute_huber_loss(const T* predictions, const T* targets, float& loss,
                        const size_t batch_size, const size_t output_size, T delta,
                        cudaStream_t stream) {
  using ComputeT = typename TypeTraits<T>::ComputePrecision;
  size_t total_size = batch_size * output_size;
  T* d_loss_values;
  cudaMallocAsync(&d_loss_values, total_size * sizeof(T), stream);

  int threads_per_block = 256;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  huber_loss_kernel<T>
      <<<num_blocks, threads_per_block>>>(predictions, targets, d_loss_values, total_size, delta);

  int block_size = 256;
  int grid_size = std::min(256, (int)((total_size + block_size - 1) / block_size));

  ComputeT* d_block_results;
  cudaMallocAsync(&d_block_results, grid_size * sizeof(ComputeT), stream);

  sum_reduce_kernel_stage1<ComputeT>
      <<<grid_size, block_size, block_size * sizeof(ComputeT), stream>>>(
          d_loss_values, d_block_results, total_size);

  ComputeT* d_total_loss;
  cudaMallocAsync(&d_total_loss, sizeof(ComputeT), stream);

  sum_reduce_kernel_stage2<ComputeT><<<1, block_size, block_size * sizeof(ComputeT), stream>>>(
      d_block_results, d_total_loss, grid_size);

  ComputeT h_total_loss;
  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(ComputeT), cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_loss_values, stream);
  cudaFreeAsync(d_block_results, stream);
  cudaFreeAsync(d_total_loss, stream);

  loss = static_cast<float>(h_total_loss) / static_cast<float>(total_size);
}

template <typename T>
void compute_huber_gradient(const T* predictions, const T* targets, T* gradient,
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

#define INSTANTIATE_LOSS_OPS(T)                                                                   \
  template void compute_crossentropy_loss<T>(const T* predictions, const T* targets, float& loss, \
                                             const size_t batch_size, const size_t num_classes,   \
                                             T epsilon, cudaStream_t stream);                     \
  template void compute_crossentropy_gradient<T>(                                                 \
      const T* predictions, const T* targets, T* gradient, const size_t batch_size,               \
      const size_t num_classes, T epsilon, cudaStream_t stream);                                  \
  template void compute_logsoftmax_crossentropy_loss<T>(                                          \
      const T* logits, const T* targets, float& loss, const size_t batch_size,                    \
      const size_t num_classes, cudaStream_t stream);                                             \
  template void compute_logsoftmax_crossentropy_gradient<T>(                                      \
      const T* logits, const T* targets, T* gradient, const size_t batch_size,                    \
      const size_t num_classes, cudaStream_t stream);                                             \
  template void compute_mse_loss<T>(const T* predictions, const T* targets, float& loss,          \
                                    const size_t batch_size, const size_t output_size,            \
                                    cudaStream_t stream);                                         \
  template void compute_mse_gradient<T>(const T* predictions, const T* targets, T* gradient,      \
                                        const size_t batch_size, const size_t output_size,        \
                                        cudaStream_t stream);                                     \
  template void compute_mae_loss<T>(const T* predictions, const T* targets, float& loss,          \
                                    const size_t batch_size, const size_t output_size,            \
                                    cudaStream_t stream);                                         \
  template void compute_mae_gradient<T>(const T* predictions, const T* targets, T* gradient,      \
                                        const size_t batch_size, const size_t output_size,        \
                                        cudaStream_t stream);                                     \
  template void compute_huber_loss<T>(const T* predictions, const T* targets, float& loss,        \
                                      const size_t batch_size, const size_t output_size, T delta, \
                                      cudaStream_t stream);                                       \
  template void compute_huber_gradient<T>(const T* predictions, const T* targets, T* gradient,    \
                                          const size_t batch_size, const size_t output_size,      \
                                          T delta, cudaStream_t stream);

INSTANTIATE_LOSS_OPS(fp16)
INSTANTIATE_LOSS_OPS(bf16)
INSTANTIATE_LOSS_OPS(float)
INSTANTIATE_LOSS_OPS(double)
#undef INSTANTIATE_LOSS_OPS

}  // namespace loss
}  // namespace cuda
}  // namespace tnn
