/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#ifdef USE_CUDNN
#include <cudnn.h>

#include <iostream>
#include <stdexcept>

#include "nn/layers_impl/cuda/cudnn_conv2d_nchw_ops.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_conv2d {

#define CHECK_CUDNN(status)                                                   \
  {                                                                           \
    if (status != CUDNN_STATUS_SUCCESS) {                                     \
      auto& err_stream = std::cerr;                                           \
      err_stream << "cuDNN Error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudnnGetErrorString(status) << std::endl;                 \
      throw std::runtime_error("cuDNN error");                                \
    }                                                                         \
  }

template <typename T>
cudnnDataType_t get_cudnn_data_type();

template <>
cudnnDataType_t get_cudnn_data_type<float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t get_cudnn_data_type<double>() {
  return CUDNN_DATA_DOUBLE;
}

ConvolutionHandle* initialize_convolution_handle(cudnnHandle_t shared_handle, size_t batch_size,
                                                 size_t in_channels, size_t input_h, size_t input_w,
                                                 size_t out_channels, size_t kernel_h,
                                                 size_t kernel_w, size_t stride_h, size_t stride_w,
                                                 size_t pad_h, size_t pad_w,
                                                 size_t workspace_limit_bytes) {
  std::cout << "Initalizing cudnn conv2d handle" << std::endl;
  ConvolutionHandle* handle = new ConvolutionHandle();

  handle->cudnn_handle = shared_handle;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->input_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->output_descriptor));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&handle->filter_descriptor));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&handle->convolution_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->bias_descriptor));
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&handle->activation_descriptor));

  size_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
  size_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->input_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, in_channels, input_h,
                                         input_w));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->output_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, out_channels, output_h,
                                         output_w));

  CHECK_CUDNN(cudnnSetFilter4dDescriptor(handle->filter_descriptor, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_h,
                                         kernel_w));

  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(handle->convolution_descriptor, pad_h, pad_w,
                                              stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->bias_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));

  int requested_algo_count = 10;
  int returned_algo_count;
  cudnnConvolutionFwdAlgoPerf_t fwd_perf[10];
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[10];
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[10];

  CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
      handle->cudnn_handle, handle->input_descriptor, handle->filter_descriptor,
      handle->convolution_descriptor, handle->output_descriptor, requested_algo_count,
      &returned_algo_count, fwd_perf));

  bool found_fwd = false;
  for (int i = 0; i < returned_algo_count; i++) {
    if (fwd_perf[i].status == CUDNN_STATUS_SUCCESS &&
        (workspace_limit_bytes == 0 || fwd_perf[i].memory <= workspace_limit_bytes)) {
      handle->fwd_algo = fwd_perf[i].algo;
      handle->fwd_workspace_size = fwd_perf[i].memory;
      found_fwd = true;

      break;
    }
  }
  if (!found_fwd) {
    throw std::runtime_error("No suitable forward convolution algorithm found within memory limit");
  }

  CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
      handle->cudnn_handle, handle->filter_descriptor, handle->output_descriptor,
      handle->convolution_descriptor, handle->input_descriptor, requested_algo_count,
      &returned_algo_count, bwd_data_perf));

  bool found_bwd_data = false;
  for (int i = 0; i < returned_algo_count; i++) {
    if (bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS &&
        (workspace_limit_bytes == 0 || bwd_data_perf[i].memory <= workspace_limit_bytes)) {
      handle->bwd_data_algo = bwd_data_perf[i].algo;
      handle->bwd_data_workspace_size = bwd_data_perf[i].memory;
      found_bwd_data = true;

      break;
    }
  }
  if (!found_bwd_data) {
    throw std::runtime_error("No suitable backward data algorithm found within memory limit");
  }

  CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
      handle->cudnn_handle, handle->input_descriptor, handle->output_descriptor,
      handle->convolution_descriptor, handle->filter_descriptor, requested_algo_count,
      &returned_algo_count, bwd_filter_perf));

  bool found_bwd_filter = false;
  for (int i = 0; i < returned_algo_count; i++) {
    if (bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS &&
        (workspace_limit_bytes == 0 || bwd_filter_perf[i].memory <= workspace_limit_bytes)) {
      handle->bwd_filter_algo = bwd_filter_perf[i].algo;
      handle->bwd_filter_workspace_size = bwd_filter_perf[i].memory;
      found_bwd_filter = true;

      break;
    }
  }
  if (!found_bwd_filter) {
    throw std::runtime_error("No suitable backward filter algorithm found within memory limit");
  }

  return handle;
}

void destroy_convolution_handle(ConvolutionHandle* handle) {
  if (handle) {
    cudnnDestroyTensorDescriptor(handle->input_descriptor);
    cudnnDestroyTensorDescriptor(handle->output_descriptor);
    cudnnDestroyFilterDescriptor(handle->filter_descriptor);
    cudnnDestroyConvolutionDescriptor(handle->convolution_descriptor);
    cudnnDestroyTensorDescriptor(handle->bias_descriptor);
    cudnnDestroyActivationDescriptor(handle->activation_descriptor);
    delete handle;
  }
}

template <typename T>
void forward_with_bias(ConvolutionHandle* handle, const void* input_data, const void* weight_data,
                       const void* bias_data, void* output_data, size_t batch_size,
                       size_t in_channels, size_t input_h, size_t input_w, size_t out_channels,
                       size_t output_h, size_t output_w, void* workspace, size_t workspace_size,
                       cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  float alpha = 1.0f;
  float beta = 0.0f;

  CHECK_CUDNN(cudnnConvolutionForward(
      handle->cudnn_handle, &alpha, handle->input_descriptor, input_data, handle->filter_descriptor,
      weight_data, handle->convolution_descriptor, handle->fwd_algo, workspace, workspace_size,
      &beta, handle->output_descriptor, output_data));

  if (bias_data) {
    alpha = 1.0f;
    beta = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(handle->cudnn_handle, &alpha, handle->bias_descriptor, bias_data,
                               &beta, handle->output_descriptor, output_data));
  }
}

template <typename T>
void backward_data(ConvolutionHandle* handle, const void* gradient_data, const void* weight_data,
                   void* input_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                   size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                   void* workspace, size_t workspace_size, cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  float alpha = T(1.0);
  float beta = T(0.0);

  CHECK_CUDNN(cudnnConvolutionBackwardData(handle->cudnn_handle, &alpha, handle->filter_descriptor,
                                           weight_data, handle->output_descriptor, gradient_data,
                                           handle->convolution_descriptor, handle->bwd_data_algo,
                                           workspace, workspace_size, &beta,
                                           handle->input_descriptor, input_grad_data));
}

template <typename T>
void backward_filter(ConvolutionHandle* handle, const void* input_data, const void* gradient_data,
                     void* weight_grad_data, size_t batch_size, size_t in_channels, size_t input_h,
                     size_t input_w, size_t out_channels, size_t output_h, size_t output_w,
                     void* workspace, size_t workspace_size, cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = T(1.0);
  T beta = T(1.0);

  CHECK_CUDNN(cudnnConvolutionBackwardFilter(
      handle->cudnn_handle, &alpha, handle->input_descriptor, input_data, handle->output_descriptor,
      gradient_data, handle->convolution_descriptor, handle->bwd_filter_algo, workspace,
      workspace_size, &beta, handle->filter_descriptor, weight_grad_data));
}

template <typename T>
void backward_bias(ConvolutionHandle* handle, const void* gradient_data, void* bias_grad_data,
                   size_t batch_size, size_t out_channels, size_t output_h, size_t output_w,
                   cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = T(1.0);
  T beta = T(1.0);

  CHECK_CUDNN(cudnnConvolutionBackwardBias(handle->cudnn_handle, &alpha, handle->output_descriptor,
                                           gradient_data, &beta, handle->bias_descriptor,
                                           bias_grad_data));
}

void update_batch_size(ConvolutionHandle* handle, size_t batch_size, size_t in_channels,
                       size_t input_h, size_t input_w, size_t out_channels, size_t output_h,
                       size_t output_w) {
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->input_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, in_channels, input_h,
                                         input_w));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->output_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batch_size, out_channels, output_h,
                                         output_w));

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
}

WorkspaceSizes get_workspace_sizes(ConvolutionHandle* handle, size_t batch_size) {
  WorkspaceSizes sizes;
  sizes.fwd_size = handle->fwd_workspace_size;
  sizes.bwd_data_size = handle->bwd_data_workspace_size;
  sizes.bwd_filter_size = handle->bwd_filter_workspace_size;
  return sizes;
}

#define INSTANTIATE_CUDNN_CONV2D(T)                                                            \
  template void forward_with_bias<T>(                                                          \
      ConvolutionHandle * handle, const void* input_data, const void* weight_data,             \
      const void* bias_data, void* output_data, size_t batch_size, size_t in_channels,         \
      size_t input_h, size_t input_w, size_t out_channels, size_t output_h, size_t output_w,   \
      void* workspace, size_t workspace_size, cudaStream_t stream);                            \
  template void backward_data<T>(                                                              \
      ConvolutionHandle * handle, const void* gradient_data, const void* weight_data,          \
      void* input_grad_data, size_t batch_size, size_t in_channels, size_t input_h,            \
      size_t input_w, size_t out_channels, size_t output_h, size_t output_w, void* workspace,  \
      size_t workspace_size, cudaStream_t stream);                                             \
  template void backward_filter<T>(                                                            \
      ConvolutionHandle * handle, const void* input_data, const void* gradient_data,           \
      void* weight_grad_data, size_t batch_size, size_t in_channels, size_t input_h,           \
      size_t input_w, size_t out_channels, size_t output_h, size_t output_w, void* workspace,  \
      size_t workspace_size, cudaStream_t stream);                                             \
  template void backward_bias<T>(ConvolutionHandle * handle, const void* gradient_data,        \
                                 void* bias_grad_data, size_t batch_size, size_t out_channels, \
                                 size_t output_h, size_t output_w, cudaStream_t stream);
INSTANTIATE_CUDNN_CONV2D(fp16)
INSTANTIATE_CUDNN_CONV2D(bf16)
INSTANTIATE_CUDNN_CONV2D(float)
INSTANTIATE_CUDNN_CONV2D(double)
#undef INSTANTIATE_CUDNN_CONV2D

}  // namespace cudnn_conv2d
}  // namespace cuda
}  // namespace tnn
#endif
