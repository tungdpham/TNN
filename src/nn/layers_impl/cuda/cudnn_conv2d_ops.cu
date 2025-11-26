/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/cudnn_conv2d_ops.hpp"

#ifdef USE_CUDNN
#include <iostream>
#include <stdexcept>

namespace tnn {
namespace cuda {
namespace cudnn_conv2d {

// Helper for cuDNN error checking
#define CHECK_CUDNN(status)                                                                        \
  {                                                                                                \
    if (status != CUDNN_STATUS_SUCCESS) {                                                          \
      std::cerr << "cuDNN Error at " << __FILE__ << ":" << __LINE__ << " - "                       \
                << cudnnGetErrorString(status) << std::endl;                                       \
      throw std::runtime_error("cuDNN error");                                                     \
    }                                                                                              \
  }

// Get cuDNN data type from template type
template <typename T> cudnnDataType_t get_cudnn_data_type();

template <> cudnnDataType_t get_cudnn_data_type<float>() { return CUDNN_DATA_FLOAT; }

template <> cudnnDataType_t get_cudnn_data_type<double>() { return CUDNN_DATA_DOUBLE; }

ConvolutionHandle *initialize_convolution_handle(size_t batch_size, size_t in_channels,
                                                 size_t input_h, size_t input_w,
                                                 size_t out_channels, size_t kernel_h,
                                                 size_t kernel_w, size_t stride_h, size_t stride_w,
                                                 size_t pad_h, size_t pad_w) {
  ConvolutionHandle *handle = new ConvolutionHandle();

  // Create cuDNN handle
  CHECK_CUDNN(cudnnCreate(&handle->cudnn_handle));

  // Create descriptors
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->input_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->output_descriptor));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&handle->filter_descriptor));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&handle->convolution_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->bias_descriptor));
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&handle->activation_descriptor));

  // Calculate output dimensions
  size_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
  size_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

  // Set input descriptor (NCHW format)
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->input_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, in_channels, input_h,
                                         input_w));

  // Set output descriptor
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->output_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, out_channels, output_h,
                                         output_w));

  // Set filter descriptor
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(handle->filter_descriptor, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_h,
                                         kernel_w));

  // Set convolution descriptor
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(handle->convolution_descriptor, pad_h, pad_w,
                                              stride_h, stride_w, 1, 1, // dilation (usually 1)
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // Set bias descriptor (1, C, 1, 1)
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->bias_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));

  // Select algorithms using the newer cuDNN 8+ API
  int requested_algo_count = 1;
  int returned_algo_count;
  cudnnConvolutionFwdAlgoPerf_t fwd_perf;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;

  // Get forward algorithm
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
      handle->cudnn_handle, handle->input_descriptor, handle->filter_descriptor,
      handle->convolution_descriptor, handle->output_descriptor, requested_algo_count,
      &returned_algo_count, &fwd_perf));
  handle->fwd_algo = fwd_perf.algo;

  // Get backward data algorithm
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
      handle->cudnn_handle, handle->filter_descriptor, handle->output_descriptor,
      handle->convolution_descriptor, handle->input_descriptor, requested_algo_count,
      &returned_algo_count, &bwd_data_perf));
  handle->bwd_data_algo = bwd_data_perf.algo;

  // Get backward filter algorithm
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      handle->cudnn_handle, handle->input_descriptor, handle->output_descriptor,
      handle->convolution_descriptor, handle->filter_descriptor, requested_algo_count,
      &returned_algo_count, &bwd_filter_perf));
  handle->bwd_filter_algo = bwd_filter_perf.algo;

  // Get workspace sizes
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      handle->cudnn_handle, handle->input_descriptor, handle->filter_descriptor,
      handle->convolution_descriptor, handle->output_descriptor, handle->fwd_algo,
      &handle->fwd_workspace_size));

  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle->cudnn_handle, handle->filter_descriptor, handle->output_descriptor,
      handle->convolution_descriptor, handle->input_descriptor, handle->bwd_data_algo,
      &handle->bwd_data_workspace_size));

  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle->cudnn_handle, handle->input_descriptor, handle->output_descriptor,
      handle->convolution_descriptor, handle->filter_descriptor, handle->bwd_filter_algo,
      &handle->bwd_filter_workspace_size));

  return handle;
}

void destroy_convolution_handle(ConvolutionHandle *handle) {
  if (handle) {
    cudnnDestroyTensorDescriptor(handle->input_descriptor);
    cudnnDestroyTensorDescriptor(handle->output_descriptor);
    cudnnDestroyFilterDescriptor(handle->filter_descriptor);
    cudnnDestroyConvolutionDescriptor(handle->convolution_descriptor);
    cudnnDestroyTensorDescriptor(handle->bias_descriptor);
    cudnnDestroyActivationDescriptor(handle->activation_descriptor);
    cudnnDestroy(handle->cudnn_handle);
    delete handle;
  }
}

template <typename T>
void forward_with_bias(ConvolutionHandle *handle, const T *input_data, const T *weight_data,
                       const T *bias_data, T *output_data, size_t batch_size, size_t in_channels,
                       size_t input_h, size_t input_w, size_t out_channels, size_t output_h,
                       size_t output_w, T *workspace, size_t workspace_size, cudaStream_t stream) {
  // Set stream for this operation
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = T(1.0);
  T beta = T(0.0);

  // Perform convolution
  CHECK_CUDNN(cudnnConvolutionForward(
      handle->cudnn_handle, &alpha, handle->input_descriptor, input_data, handle->filter_descriptor,
      weight_data, handle->convolution_descriptor, handle->fwd_algo, workspace, workspace_size,
      &beta, handle->output_descriptor, output_data));

  // Add bias if provided
  if (bias_data) {
    alpha = T(1.0);
    beta = T(1.0);
    CHECK_CUDNN(cudnnAddTensor(handle->cudnn_handle, &alpha, handle->bias_descriptor, bias_data,
                               &beta, handle->output_descriptor, output_data));
  }
}

template <typename T>
void backward_data(ConvolutionHandle *handle, const T *gradient_data, const T *weight_data,
                   T *input_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                   size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                   T *workspace, size_t workspace_size, cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = T(1.0);
  T beta = T(0.0);

  CHECK_CUDNN(cudnnConvolutionBackwardData(handle->cudnn_handle, &alpha, handle->filter_descriptor,
                                           weight_data, handle->output_descriptor, gradient_data,
                                           handle->convolution_descriptor, handle->bwd_data_algo,
                                           workspace, workspace_size, &beta,
                                           handle->input_descriptor, input_grad_data));
}

template <typename T>
void backward_filter(ConvolutionHandle *handle, const T *input_data, const T *gradient_data,
                     T *weight_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                     size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                     T *workspace, size_t workspace_size, cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = T(1.0);
  T beta = T(1.0); // Accumulate gradients

  CHECK_CUDNN(cudnnConvolutionBackwardFilter(
      handle->cudnn_handle, &alpha, handle->input_descriptor, input_data, handle->output_descriptor,
      gradient_data, handle->convolution_descriptor, handle->bwd_filter_algo, workspace,
      workspace_size, &beta, handle->filter_descriptor, weight_grad_data));
}

template <typename T>
void backward_bias(ConvolutionHandle *handle, const T *gradient_data, T *bias_grad_data,
                   size_t batch_size, size_t out_channels, size_t output_h, size_t output_w,
                   cudaStream_t stream) {
  // Reuse existing cuDNN handle and descriptors to avoid implicit synchronization
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = T(1.0);
  T beta = T(1.0); // Accumulate gradients

  CHECK_CUDNN(cudnnConvolutionBackwardBias(handle->cudnn_handle, &alpha, handle->output_descriptor,
                                           gradient_data, &beta, handle->bias_descriptor,
                                           bias_grad_data));
}

void update_batch_size(ConvolutionHandle *handle, size_t batch_size, size_t in_channels,
                       size_t input_h, size_t input_w, size_t out_channels, size_t output_h,
                       size_t output_w) {
  // Update input descriptor for new batch size
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->input_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, in_channels, input_h,
                                         input_w));

  // Update output descriptor for new batch size
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->output_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, out_channels, output_h,
                                         output_w));
}

WorkspaceSizes get_workspace_sizes(ConvolutionHandle *handle, size_t batch_size) {
  WorkspaceSizes sizes;
  sizes.fwd_size = handle->fwd_workspace_size;
  sizes.bwd_data_size = handle->bwd_data_workspace_size;
  sizes.bwd_filter_size = handle->bwd_filter_workspace_size;
  return sizes;
}

// Explicit template instantiations
template void forward_with_bias<float>(ConvolutionHandle *handle, const float *input_data,
                                       const float *weight_data, const float *bias_data,
                                       float *output_data, size_t batch_size, size_t in_channels,
                                       size_t input_h, size_t input_w, size_t out_channels,
                                       size_t output_h, size_t output_w, float *workspace,
                                       size_t workspace_size, cudaStream_t stream);

template void backward_data<float>(ConvolutionHandle *handle, const float *gradient_data,
                                   const float *weight_data, float *input_grad_data,
                                   size_t batch_size, size_t in_channels, size_t input_h,
                                   size_t input_w, size_t out_channels, size_t output_h,
                                   size_t output_w, float *workspace, size_t workspace_size,
                                   cudaStream_t stream);

template void backward_filter<float>(ConvolutionHandle *handle, const float *input_data,
                                     const float *gradient_data, float *weight_grad_data,
                                     size_t batch_size, size_t in_channels, size_t input_h,
                                     size_t input_w, size_t out_channels, size_t output_h,
                                     size_t output_w, float *workspace, size_t workspace_size,
                                     cudaStream_t stream);

template void backward_bias<float>(ConvolutionHandle *handle, const float *gradient_data,
                                   float *bias_grad_data, size_t batch_size, size_t out_channels,
                                   size_t output_h, size_t output_w, cudaStream_t stream);

} // namespace cudnn_conv2d
} // namespace cuda
} // namespace tnn
#endif
