/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "utils/accuracy_impl/cuda/accuracy.hpp"
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace accuracy {

__global__ void compute_class_accuracy_kernel(const float *predictions, const float *targets,
                                              int *correct_count, const size_t batch_size,
                                              const size_t num_classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  // Find predicted class
  int pred_class = 0;
  float max_pred = predictions[idx * num_classes];
  for (size_t j = 1; j < num_classes; ++j) {
    const float pred_val = predictions[idx * num_classes + j];
    if (pred_val > max_pred) {
      max_pred = pred_val;
      pred_class = static_cast<int>(j);
    }
  }

  // Find true class
  int true_class = -1;
  for (size_t j = 0; j < num_classes; ++j) {
    if (targets[idx * num_classes + j] > 0.5f) {
      true_class = static_cast<int>(j);
      break;
    }
  }

  // Increment counter if correct
  if (pred_class == true_class && true_class != -1) {
    atomicAdd(correct_count, 1);
  }
}

__global__ void compute_class_corrects_kernel(const float *predictions, const float *targets,
                                              int *correct_count, const size_t batch_size,
                                              const size_t num_classes, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  // Find predicted class
  int pred_class = 0;
  float max_pred = predictions[idx * num_classes];
  for (size_t j = 1; j < num_classes; ++j) {
    const float pred_val = predictions[idx * num_classes + j];
    if (pred_val > max_pred) {
      max_pred = pred_val;
      pred_class = static_cast<int>(j);
    }
  }

  // Find true class
  int true_class = -1;
  for (size_t j = 0; j < num_classes; ++j) {
    if (targets[idx * num_classes + j] > threshold) {
      true_class = static_cast<int>(j);
      break;
    }
  }

  // Increment counter if correct
  if (pred_class == true_class && true_class != -1) {
    atomicAdd(correct_count, 1);
  }
}

float compute_class_accuracy(const float *predictions, const float *targets,
                             const size_t batch_size, const size_t num_classes) {
  int *d_correct_count;
  cudaMalloc(&d_correct_count, sizeof(int));
  cudaMemset(d_correct_count, 0, sizeof(int));

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_class_accuracy_kernel<<<grid_size, block_size>>>(predictions, targets, d_correct_count,
                                                           batch_size, num_classes);

  int h_correct_count;
  cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_correct_count);

  return static_cast<float>(h_correct_count) / static_cast<float>(batch_size);
}

float compute_class_corrects(const float *predictions, const float *targets,
                             const size_t batch_size, const size_t num_classes, float threshold) {
  int *d_correct_count;
  cudaMalloc(&d_correct_count, sizeof(int));
  cudaMemset(d_correct_count, 0, sizeof(int));

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_class_corrects_kernel<<<grid_size, block_size>>>(predictions, targets, d_correct_count,
                                                           batch_size, num_classes, threshold);

  int h_correct_count;
  cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_correct_count);

  return static_cast<float>(h_correct_count);
}

} // namespace accuracy
} // namespace cuda
} // namespace tnn
