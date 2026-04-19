/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <cuda_runtime.h>

#include "nn/metrics_impl/cuda/metrics.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace metrics {

// Helper kernel for precision/recall computation
template <typename T>
__global__ void compute_precision_recall_kernel(const T* __restrict__ predictions,
                                                const int* __restrict__ targets,
                                                int* __restrict__ tp, int* __restrict__ fp,
                                                int* __restrict__ fn, const size_t batch_size,
                                                const size_t num_classes, int class_id) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_tp = 0, local_fp = 0, local_fn = 0;

  if (idx < batch_size) {
    int pred_class = 0;
    float max_pred = static_cast<float>(predictions[idx * num_classes]);
    for (size_t j = 1; j < num_classes; ++j) {
      float pred_val = static_cast<float>(predictions[idx * num_classes + j]);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = targets[idx];

    // True Positives
    if (pred_class == class_id && true_class == class_id) {
      local_tp = 1;
    }
    // False Positives
    if (pred_class == class_id && true_class != class_id) {
      local_fp = 1;
    }
    // False Negatives
    if (pred_class != class_id && true_class == class_id) {
      local_fn = 1;
    }
  }

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    local_tp += __shfl_down_sync(0xFFFFFFFF, local_tp, offset);
    local_fp += __shfl_down_sync(0xFFFFFFFF, local_fp, offset);
    local_fn += __shfl_down_sync(0xFFFFFFFF, local_fn, offset);
  }

  __shared__ int shared_tp[32], shared_fp[32], shared_fn[32];
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane == 0) {
    shared_tp[warp_id] = local_tp;
    shared_fp[warp_id] = local_fp;
    shared_fn[warp_id] = local_fn;
  }

  __syncthreads();

  // Final reduction
  if (warp_id == 0) {
    int block_tp = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_tp[lane] : 0;
    int block_fp = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_fp[lane] : 0;
    int block_fn = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_fn[lane] : 0;

    for (int offset = 16; offset > 0; offset /= 2) {
      block_tp += __shfl_down_sync(0xFFFFFFFF, block_tp, offset);
      block_fp += __shfl_down_sync(0xFFFFFFFF, block_fp, offset);
      block_fn += __shfl_down_sync(0xFFFFFFFF, block_fn, offset);
    }

    if (lane == 0) {
      atomicAdd(tp, block_tp);
      atomicAdd(fp, block_fp);
      atomicAdd(fn, block_fn);
    }
  }
}

template <typename T>
float compute_precision(const T* predictions, const int* targets, const size_t batch_size,
                        const size_t num_classes, int class_id, cudaStream_t stream) {
  if (class_id == -1) {
    // Macro-average precision across all classes
    float total_precision = 0.0f;
    int valid_classes = 0;

    for (int c = 0; c < static_cast<int>(num_classes); ++c) {
      float class_precision =
          compute_precision(predictions, targets, batch_size, num_classes, c, stream);
      if (!std::isnan(class_precision)) {
        total_precision += class_precision;
        valid_classes++;
      }
    }

    return valid_classes > 0 ? total_precision / valid_classes : 0.0f;
  }

  int *d_tp, *d_fp, *d_fn;
  cudaMalloc(&d_tp, sizeof(int));
  cudaMalloc(&d_fp, sizeof(int));
  cudaMalloc(&d_fn, sizeof(int));
  cudaMemsetAsync(d_tp, 0, sizeof(int), stream);
  cudaMemsetAsync(d_fp, 0, sizeof(int), stream);
  cudaMemsetAsync(d_fn, 0, sizeof(int), stream);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_precision_recall_kernel<T><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, d_tp, d_fp, d_fn, batch_size, num_classes, class_id);

  int h_tp = 0, h_fp = 0;
  cudaMemcpyAsync(&h_tp, d_tp, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&h_fp, d_fp, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_tp);
  cudaFree(d_fp);
  cudaFree(d_fn);

  int total_predicted_positive = h_tp + h_fp;
  return total_predicted_positive > 0 ? static_cast<float>(h_tp) / total_predicted_positive : 0.0f;
}

template <typename T>
float compute_recall(const T* predictions, const int* targets, const size_t batch_size,
                     const size_t num_classes, int class_id, cudaStream_t stream) {
  if (class_id == -1) {
    // Macro-average recall across all classes
    float total_recall = 0.0f;
    int valid_classes = 0;

    for (int c = 0; c < static_cast<int>(num_classes); ++c) {
      float class_recall = compute_recall(predictions, targets, batch_size, num_classes, c, stream);
      if (!std::isnan(class_recall)) {
        total_recall += class_recall;
        valid_classes++;
      }
    }

    return valid_classes > 0 ? total_recall / valid_classes : 0.0f;
  }

  int *d_tp, *d_fp, *d_fn;
  cudaMalloc(&d_tp, sizeof(int));
  cudaMalloc(&d_fp, sizeof(int));
  cudaMalloc(&d_fn, sizeof(int));
  cudaMemsetAsync(d_tp, 0, sizeof(int), stream);
  cudaMemsetAsync(d_fp, 0, sizeof(int), stream);
  cudaMemsetAsync(d_fn, 0, sizeof(int), stream);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_precision_recall_kernel<T><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, d_tp, d_fp, d_fn, batch_size, num_classes, class_id);

  int h_tp = 0, h_fn = 0;
  cudaMemcpyAsync(&h_tp, d_tp, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&h_fn, d_fn, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_tp);
  cudaFree(d_fp);
  cudaFree(d_fn);

  int total_actual_positive = h_tp + h_fn;
  return total_actual_positive > 0 ? static_cast<float>(h_tp) / total_actual_positive : 0.0f;
}

template <typename T>
float compute_f1_score(const T* predictions, const int* targets, const size_t batch_size,
                       const size_t num_classes, int class_id, cudaStream_t stream) {
  float precision =
      compute_precision(predictions, targets, batch_size, num_classes, class_id, stream);
  float recall = compute_recall(predictions, targets, batch_size, num_classes, class_id, stream);

  if (precision + recall > 0.0f) {
    return 2.0f * (precision * recall) / (precision + recall);
  }
  return 0.0f;
}

// Perplexity kernel
template <typename T>
__global__ void compute_perplexity_kernel(const T* __restrict__ predictions,
                                          const int* __restrict__ targets,
                                          double* __restrict__ log_likelihood,
                                          const size_t batch_size, const size_t num_classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double local_ll = 0.0;
  const double epsilon = 1e-10;

  if (idx < batch_size) {
    int true_class = targets[idx];
    double prob = static_cast<double>(predictions[idx * num_classes + true_class]);
    prob = max(prob, epsilon);
    local_ll = log(prob);
  }

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    local_ll += __shfl_down_sync(0xFFFFFFFF, local_ll, offset);
  }

  __shared__ double shared_ll[32];
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane == 0) shared_ll[warp_id] = local_ll;

  __syncthreads();

  if (warp_id == 0) {
    double block_ll = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_ll[lane] : 0.0;
    for (int offset = 16; offset > 0; offset /= 2) {
      block_ll += __shfl_down_sync(0xFFFFFFFF, block_ll, offset);
    }

    if (lane == 0) {
      atomicAdd(log_likelihood, block_ll);
    }
  }
}

template <typename T>
float compute_perplexity(const T* predictions, const int* targets, const size_t batch_size,
                         const size_t num_classes, cudaStream_t stream) {
  double* d_log_likelihood;
  cudaMalloc(&d_log_likelihood, sizeof(double));
  cudaMemsetAsync(d_log_likelihood, 0, sizeof(double), stream);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_perplexity_kernel<T><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, d_log_likelihood, batch_size, num_classes);

  double h_log_likelihood = 0.0;
  cudaMemcpyAsync(&h_log_likelihood, d_log_likelihood, sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_log_likelihood);

  double avg_log_likelihood = h_log_likelihood / static_cast<double>(batch_size);
  return static_cast<float>(exp(-avg_log_likelihood));
}

// Top-K accuracy kernel
template <typename T>
__global__ void compute_top_k_accuracy_kernel(const T* __restrict__ predictions,
                                              const int* __restrict__ targets,
                                              int* __restrict__ correct_count,
                                              const size_t batch_size, const size_t num_classes,
                                              int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_hit = 0;

  if (idx < batch_size) {
    int true_class = targets[idx];

    // Simple bubble sort for top-k (works for small k)
    // For larger k, consider a more efficient algorithm
    float top_k_vals[32];  // Assumes k <= 32
    int top_k_indices[32];

    // Initialize with first k values
    int actual_k = min(k, static_cast<int>(num_classes));
    for (int i = 0; i < actual_k; ++i) {
      top_k_vals[i] = static_cast<float>(predictions[idx * num_classes + i]);
      top_k_indices[i] = i;
    }

    // Sort initial k elements
    for (int i = 0; i < actual_k; ++i) {
      for (int j = i + 1; j < actual_k; ++j) {
        if (top_k_vals[j] > top_k_vals[i]) {
          float tmp_val = top_k_vals[i];
          int tmp_idx = top_k_indices[i];
          top_k_vals[i] = top_k_vals[j];
          top_k_indices[i] = top_k_indices[j];
          top_k_vals[j] = tmp_val;
          top_k_indices[j] = tmp_idx;
        }
      }
    }

    // Check remaining elements
    for (int i = actual_k; i < num_classes; ++i) {
      float val = static_cast<float>(predictions[idx * num_classes + i]);
      if (val > top_k_vals[actual_k - 1]) {
        // Insert in sorted position
        int insert_pos = actual_k - 1;
        for (int j = 0; j < actual_k - 1; ++j) {
          if (val > top_k_vals[j]) {
            insert_pos = j;
            break;
          }
        }
        // Shift elements
        for (int j = actual_k - 1; j > insert_pos; --j) {
          top_k_vals[j] = top_k_vals[j - 1];
          top_k_indices[j] = top_k_indices[j - 1];
        }
        top_k_vals[insert_pos] = val;
        top_k_indices[insert_pos] = i;
      }
    }

    // Check if true class is in top-k
    for (int i = 0; i < actual_k; ++i) {
      if (top_k_indices[i] == true_class) {
        local_hit = 1;
        break;
      }
    }
  }

  // Warp-level reduction
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
      atomicAdd(correct_count, block_sum);
    }
  }
}

template <typename T>
float compute_top_k_accuracy(const T* predictions, const int* targets, const size_t batch_size,
                             const size_t num_classes, int k, cudaStream_t stream) {
  int* d_correct_count;
  cudaMalloc(&d_correct_count, sizeof(int));
  cudaMemsetAsync(d_correct_count, 0, sizeof(int), stream);

  int block_size = 256;
  int grid_size = (batch_size + block_size - 1) / block_size;

  compute_top_k_accuracy_kernel<T><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, d_correct_count, batch_size, num_classes, k);

  int h_correct_count = 0;
  cudaMemcpyAsync(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_correct_count);

  return static_cast<float>(h_correct_count) / static_cast<float>(batch_size);
}

// MAE kernel
template <typename T>
__global__ void compute_mae_kernel(const T* __restrict__ predictions, const T* __restrict__ targets,
                                   double* __restrict__ total_error, const size_t total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double local_error = 0.0;

  if (idx < total_elements) {
    double diff = static_cast<double>(predictions[idx]) - static_cast<double>(targets[idx]);
    local_error = abs(diff);
  }

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    local_error += __shfl_down_sync(0xFFFFFFFF, local_error, offset);
  }

  __shared__ double shared_errors[32];
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane == 0) shared_errors[warp_id] = local_error;

  __syncthreads();

  if (warp_id == 0) {
    double block_error = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_errors[lane] : 0.0;
    for (int offset = 16; offset > 0; offset /= 2) {
      block_error += __shfl_down_sync(0xFFFFFFFF, block_error, offset);
    }

    if (lane == 0) {
      atomicAdd(total_error, block_error);
    }
  }
}

template <typename T>
float compute_mae(const T* predictions, const T* targets, const size_t total_elements,
                  cudaStream_t stream) {
  double* d_total_error;
  cudaMalloc(&d_total_error, sizeof(double));
  cudaMemsetAsync(d_total_error, 0, sizeof(double), stream);

  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;

  compute_mae_kernel<T>
      <<<grid_size, block_size, 0, stream>>>(predictions, targets, d_total_error, total_elements);

  double h_total_error = 0.0;
  cudaMemcpyAsync(&h_total_error, d_total_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_total_error);

  return static_cast<float>(h_total_error / static_cast<double>(total_elements));
}

// MSE kernel
template <typename T>
__global__ void compute_mse_kernel(const T* __restrict__ predictions, const T* __restrict__ targets,
                                   double* __restrict__ total_squared_error,
                                   const size_t total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double local_error = 0.0;

  if (idx < total_elements) {
    double diff = static_cast<double>(predictions[idx]) - static_cast<double>(targets[idx]);
    local_error = diff * diff;
  }

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    local_error += __shfl_down_sync(0xFFFFFFFF, local_error, offset);
  }

  __shared__ double shared_errors[32];
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane == 0) shared_errors[warp_id] = local_error;

  __syncthreads();

  if (warp_id == 0) {
    double block_error = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_errors[lane] : 0.0;
    for (int offset = 16; offset > 0; offset /= 2) {
      block_error += __shfl_down_sync(0xFFFFFFFF, block_error, offset);
    }

    if (lane == 0) {
      atomicAdd(total_squared_error, block_error);
    }
  }
}

template <typename T>
float compute_mse(const T* predictions, const T* targets, const size_t total_elements,
                  cudaStream_t stream) {
  double* d_total_squared_error;
  cudaMalloc(&d_total_squared_error, sizeof(double));
  cudaMemsetAsync(d_total_squared_error, 0, sizeof(double), stream);

  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;

  compute_mse_kernel<T><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, d_total_squared_error, total_elements);

  double h_total_squared_error = 0.0;
  cudaMemcpyAsync(&h_total_squared_error, d_total_squared_error, sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_total_squared_error);

  return static_cast<float>(h_total_squared_error / static_cast<double>(total_elements));
}

// Class corrects kernel
template <typename T>
__global__ void compute_class_corrects_kernel(const T* __restrict__ predictions,
                                              const int* __restrict__ targets,
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

    int true_class = targets[idx];

    if (pred_class == true_class) {
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
int compute_class_corrects(const T* predictions, const int* targets, const size_t batch_size,
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

#define INSTANTIATE(T)                                                                        \
  template float compute_precision<T>(const T* predictions, const int* targets,               \
                                      const size_t batch_size, const size_t num_classes,      \
                                      int class_id, cudaStream_t stream);                     \
  template float compute_recall<T>(const T* predictions, const int* targets,                  \
                                   const size_t batch_size, const size_t num_classes,         \
                                   int class_id, cudaStream_t stream);                        \
  template float compute_f1_score<T>(const T* predictions, const int* targets,                \
                                     const size_t batch_size, const size_t num_classes,       \
                                     int class_id, cudaStream_t stream);                      \
  template float compute_perplexity<T>(const T* predictions, const int* targets,              \
                                       const size_t batch_size, const size_t num_classes,     \
                                       cudaStream_t stream);                                  \
  template float compute_top_k_accuracy<T>(const T* predictions, const int* targets,          \
                                           const size_t batch_size, const size_t num_classes, \
                                           int k, cudaStream_t stream);                       \
  template float compute_mae<T>(const T* predictions, const T* targets,                       \
                                const size_t total_elements, cudaStream_t stream);            \
  template float compute_mse<T>(const T* predictions, const T* targets,                       \
                                const size_t total_elements, cudaStream_t stream);            \
  template int compute_class_corrects<T>(const T* predictions, const int* targets,            \
                                         const size_t batch_size, const size_t num_classes,   \
                                         float threshold, cudaStream_t stream);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace metrics
}  // namespace cuda
}  // namespace tnn
