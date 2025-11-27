/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDNN
#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace tnn {
namespace cuda {
namespace cudnn_conv2d {

// Opaque handle for cuDNN descriptors stored per Conv2D instance
struct ConvolutionHandle {
  cudnnHandle_t cudnn_handle;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t filter_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnTensorDescriptor_t bias_descriptor;
  cudnnActivationDescriptor_t activation_descriptor;

  // Algorithm and workspace info
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  size_t fwd_workspace_size;
  size_t bwd_data_workspace_size;
  size_t bwd_filter_workspace_size;
};

// Initialize cuDNN handle and descriptors for a conv layer
// workspace_limit_bytes: maximum workspace size per algorithm (0 = no limit)
//   Recommended values: 32MB (balanced), 16MB (conservative), 8MB (minimal)
ConvolutionHandle *initialize_convolution_handle(size_t batch_size, size_t in_channels,
                                                 size_t input_h, size_t input_w,
                                                 size_t out_channels, size_t kernel_h,
                                                 size_t kernel_w, size_t stride_h, size_t stride_w,
                                                 size_t pad_h, size_t pad_w,
                                                 size_t workspace_limit_bytes = 128 * 1024 * 1024);

// Destroy cuDNN handle and clean up descriptors
void destroy_convolution_handle(ConvolutionHandle *handle);

// Update tensor descriptors for a different batch size (avoids full reinitialization)
void update_batch_size(ConvolutionHandle *handle, size_t batch_size, size_t in_channels,
                       size_t input_h, size_t input_w, size_t out_channels, size_t output_h,
                       size_t output_w);

// Forward pass using cuDNN
template <typename T>
void forward_with_bias(ConvolutionHandle *handle, const T *input_data, const T *weight_data,
                       const T *bias_data, T *output_data, size_t batch_size, size_t in_channels,
                       size_t input_h, size_t input_w, size_t out_channels, size_t output_h,
                       size_t output_w, T *workspace, size_t workspace_size, cudaStream_t stream);

// Backward pass - input gradients
template <typename T>
void backward_data(ConvolutionHandle *handle, const T *gradient_data, const T *weight_data,
                   T *input_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                   size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                   T *workspace, size_t workspace_size, cudaStream_t stream);

// Backward pass - weight gradients
template <typename T>
void backward_filter(ConvolutionHandle *handle, const T *input_data, const T *gradient_data,
                     T *weight_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                     size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                     T *workspace, size_t workspace_size, cudaStream_t stream);

// Backward pass - bias gradients (reduction kernel)
template <typename T>
void backward_bias(ConvolutionHandle *handle, const T *gradient_data, T *bias_grad_data,
                   size_t batch_size, size_t out_channels, size_t output_h, size_t output_w,
                   cudaStream_t stream);

// Get required workspace sizes
struct WorkspaceSizes {
  size_t fwd_size;
  size_t bwd_data_size;
  size_t bwd_filter_size;
};

WorkspaceSizes get_workspace_sizes(ConvolutionHandle *handle, size_t batch_size);

} // namespace cudnn_conv2d
} // namespace cuda
} // namespace tnn
#endif
