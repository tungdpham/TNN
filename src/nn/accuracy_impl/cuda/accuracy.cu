/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/accuracy_impl/cuda/accuracy.hpp"
#include "type/type.hpp"
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace accuracy {

template <typename T>
__global__ void compute_class_accuracy_kernel(const T *predictions, const T *targets,
                                              int *correct_count, const size_t batch_size,
                                              const size_t num_classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  int pred_class = 0;
  float max_pred = static_cast<float>(predictions[idx * num_classes]);
  for (size_t j = 1; j < num_classes; ++j) {
    const float pred_val = static_cast<float>(predictions[idx * num_classes + j]);
    if (pred_val > max_pred) {
      max_pred = pred_val;
      pred_class = static_cast<int>(j);
    }
  }

  int true_class = -1;
  for (size_t j = 0; j < num_classes; ++j) {
    if (static_cast<float>(targets[idx * num_classes + j]) > 0.5f) {
      true_class = static_cast<int>(j);
      break;
    }
  }

  if (pred_class == true_class && true_class != -1) {
    atomicAdd(correct_count, 1);
  }
}

template <typename T>
__global__ void compute_class_corrects_kernel(const T *predictions, const T *targets,
                                              int *correct_count, const size_t batch_size,
                                              const size_t num_classes, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  int pred_class = 0;
  float max_pred = static_cast<float>(predictions[idx * num_classes]);
  for (size_t j = 1; j < num_classes; ++j) {
    const float pred_val = static_cast<float>(predictions[idx * num_classes + j]);
    if (pred_val > max_pred) {
      max_pred = pred_val;
      pred_class = static_cast<int>(j);
    }
  }

  int true_class = -1;
  for (size_t j = 0; j < num_classes; ++j) {
    if (static_cast<float>(targets[idx * num_classes + j]) > threshold) {
      true_class = static_cast<int>(j);
      break;
    }
  }

  if (pred_class == true_class && true_class != -1) {
    atomicAdd(correct_count, 1);
  }
}

template <typename T>
float compute_class_accuracy(const T *predictions, const T *targets, const size_t batch_size,
                             const size_t num_classes) {
  int *d_correct_count;
  cudaMalloc(&d_correct_count, sizeof(int));
  cudaMemset(d_correct_count, 0, sizeof(int));

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_class_accuracy_kernel<T>
      <<<grid_size, block_size>>>(predictions, targets, d_correct_count, batch_size, num_classes);

  int h_correct_count;
  cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_correct_count);

  return static_cast<float>(h_correct_count) / static_cast<float>(batch_size);
}

template <typename T>
int compute_class_corrects(const T *predictions, const T *targets, const size_t batch_size,
                           const size_t num_classes, float threshold) {
  int *d_correct_count;
  cudaMalloc(&d_correct_count, sizeof(int));
  cudaMemset(d_correct_count, 0, sizeof(int));

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_class_corrects_kernel<T><<<grid_size, block_size>>>(predictions, targets, d_correct_count,
                                                              batch_size, num_classes, threshold);

  int h_correct_count;
  cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_correct_count);

  return h_correct_count;
}

template float compute_class_accuracy<float>(const float *, const float *, const size_t,
                                             const size_t);
template float compute_class_accuracy<double>(const double *, const double *, const size_t,
                                              const size_t);
template float compute_class_accuracy<fp16>(const fp16 *, const fp16 *, const size_t, const size_t);
template float compute_class_accuracy<bf16>(const bf16 *, const bf16 *, const size_t, const size_t);

template int compute_class_corrects<float>(const float *, const float *, const size_t, const size_t,
                                           float);
template int compute_class_corrects<double>(const double *, const double *, const size_t,
                                            const size_t, float);
template int compute_class_corrects<fp16>(const fp16 *, const fp16 *, const size_t, const size_t,
                                          float);
template int compute_class_corrects<bf16>(const bf16 *, const bf16 *, const size_t, const size_t,
                                          float);

} // namespace accuracy
} // namespace cuda
} // namespace tnn
