/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <cuda_runtime.h>

#include "nn/accuracy_impl/cuda/accuracy.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace accuracy {

template <typename T>
__global__ void compute_class_corrects_kernel(const T* __restrict__ predictions,
                                              const T* __restrict__ targets,
                                              int* __restrict__ global_correct_count,
                                              const size_t batch_size, const size_t num_classes,
                                              float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_hit = 0;

  if (idx < batch_size) {
    int pred_class = 0;
    float max_pred = static_cast<float>(predictions[idx * num_classes]);
    for (size_t j = 1; j < num_classes; ++j) {
      float pred_val = static_cast<float>(predictions[idx * num_classes + j]);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = (int)j;
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (static_cast<float>(targets[idx * num_classes + j]) > threshold) {
        true_class = (int)j;
        break;
      }
    }

    if (pred_class == true_class && true_class != -1) {
      local_hit = 1;
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    local_hit += __shfl_down_sync(0xFFFFFFFF, local_hit, offset);
  }

  __shared__ int shared_sums[32];
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane == 0) shared_sums[warp_id] = local_hit;

  __syncthreads();

  if (warp_id == 0) {
    int block_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sums[lane] : 0;
    for (int offset = 16; offset > 0; offset /= 2) {
      block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
    }

    if (lane == 0) {
      atomicAdd(global_correct_count, block_sum);
    }
  }
}

template <typename T>
int compute_class_corrects(const T* predictions, const T* targets, const size_t batch_size,
                           const size_t num_classes, float threshold, cudaStream_t stream) {
  int* d_correct_count;
  cudaMalloc(&d_correct_count, sizeof(int));
  cudaMemsetAsync(d_correct_count, 0, sizeof(int), stream);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_class_corrects_kernel<T><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, d_correct_count, batch_size, num_classes, threshold);

  int h_correct_count = 0;
  cudaMemcpyAsync(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  cudaFree(d_correct_count);

  return h_correct_count;
}

#define INSTANTIATE_COMPUTE_CLASS_CORRECTS(T)                                               \
  template int compute_class_corrects<T>(const T* predictions, const T* targets,            \
                                         const size_t batch_size, const size_t num_classes, \
                                         float threshold, cudaStream_t stream);
INSTANTIATE_COMPUTE_CLASS_CORRECTS(bf16)
INSTANTIATE_COMPUTE_CLASS_CORRECTS(fp16)
INSTANTIATE_COMPUTE_CLASS_CORRECTS(float)
INSTANTIATE_COMPUTE_CLASS_CORRECTS(double)
#undef INSTANTIATE_COMPUTE_CLASS_CORRECTS

}  // namespace accuracy
}  // namespace cuda
}  // namespace tnn
