/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/common/conv2d.hpp"
#ifdef USE_CUDNN
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>

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
};

// Initialize cuDNN handle and descriptors for a conv layer, with optional workspace size limit
// Recommended values: 256MB (Performance), 64MB (conservative), 32MB (minimal)
ConvolutionHandle *initialize_convolution_handle(cudnnHandle_t shared_handle,
                                                 ConvolutionStats &stats,
                                                 size_t workspace_limit_bytes = 256 * 1024 * 1024);

// Destroy cuDNN handle and clean up descriptors
void destroy_convolution_handle(ConvolutionHandle *handle);

template <typename T>
void forward_with_bias(ConvolutionHandle *handle, const void *input_data, const void *weight_data,
                       const void *bias_data, void *output_data, size_t batch_size,
                       size_t in_channels, size_t input_h, size_t input_w, size_t out_channels,
                       size_t output_h, size_t output_w, void *workspace, size_t workspace_size,
                       cudaStream_t stream);

template <typename T>
void run_dgrad(ConvolutionHandle *handle, const void *gradient_data, const void *weight_data,
               void *input_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
               size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
               void *workspace, size_t workspace_size, cudaStream_t stream);

template <typename T>
void backward_filter(ConvolutionHandle *handle, const void *input_data, const void *gradient_data,
                     void *weight_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                     size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                     void *workspace, size_t workspace_size, cudaStream_t stream);

template <typename T>
void run_bgrad(ConvolutionHandle *handle, const void *gradient_data, void *bias_grad_data,
               size_t batch_size, size_t out_channels, size_t output_h, size_t output_w,
               cudaStream_t stream);
}  // namespace cudnn_conv2d
}  // namespace cuda
}  // namespace tnn
#endif
