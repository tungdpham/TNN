/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDNN
#include <cuda_runtime.h>
#include <cudnn.h>

namespace tnn {
namespace cuda {
namespace cudnn_batchnorm {

struct BatchNormHandle {
  cudnnHandle_t cudnn_handle;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnTensorDescriptor_t bn_scale_bias_mean_var_descriptor;
};

template <typename T> cudnnDataType_t get_cudnn_data_type();

BatchNormHandle *initialize_batchnorm_handle(cudnnHandle_t shared_handle, size_t batch_size,
                                             size_t channels, size_t spatial_size,
                                             cudnnDataType_t data_type);
void destroy_batchnorm_handle(BatchNormHandle *handle);

template <typename T>
void run_forward_training(BatchNormHandle *handle, const T *input, const T *bnScale,
                          const T *bnBias, T *output, T *running_mean, T *running_var, T *save_mean,
                          T *save_inv_var, double epsilon, double exponential_average_factor,
                          cudaStream_t stream);

template <typename T>
void run_forward_inference(BatchNormHandle *handle, const T *input, const T *bnScale,
                           const T *bnBias, const T *estimated_mean, const T *estimated_var,
                           T *output, double epsilon, cudaStream_t stream);

template <typename T>
void run_backward(BatchNormHandle *handle, const T *input, const T *grad_output, const T *bnScale,
                  T *grad_input, T *grad_bnScale, T *grad_bnBias, const T *save_mean,
                  const T *save_inv_var, double epsilon, cudaStream_t stream);

} // namespace cudnn_batchnorm
} // namespace cuda
} // namespace tnn
#endif
