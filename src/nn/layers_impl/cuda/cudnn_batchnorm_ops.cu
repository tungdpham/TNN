/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/cudnn_batchnorm_ops.hpp"
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDNN

namespace tnn {
namespace cuda {
namespace cudnn_batchnorm {

#define CHECK_CUDNN(status)                                                                        \
  {                                                                                                \
    if (status != CUDNN_STATUS_SUCCESS) {                                                          \
      std::cerr << "cuDNN Error at " << __FILE__ << ":" << __LINE__ << " - "                       \
                << cudnnGetErrorString(status) << std::endl;                                       \
      throw std::runtime_error("cuDNN error");                                                     \
    }                                                                                              \
  }

template <typename T> cudnnDataType_t get_cudnn_data_type();

template <> cudnnDataType_t get_cudnn_data_type<float>() { return CUDNN_DATA_FLOAT; }
template <> cudnnDataType_t get_cudnn_data_type<double>() { return CUDNN_DATA_DOUBLE; }

BatchNormHandle *initialize_batchnorm_handle(cudnnHandle_t shared_handle, size_t N, size_t C,
                                             size_t H, size_t W, cudnnDataType_t data_type) {
  BatchNormHandle *handle = new BatchNormHandle();
  handle->cudnn_handle = shared_handle;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->input_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->output_descriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&handle->bn_scale_bias_mean_var_descriptor));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->input_descriptor, CUDNN_TENSOR_NCHW, data_type, N,
                                         C, H, W));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(handle->output_descriptor, CUDNN_TENSOR_NCHW, data_type, N,
                                         C, H, W));

  CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(handle->bn_scale_bias_mean_var_descriptor,
                                            handle->input_descriptor, CUDNN_BATCHNORM_SPATIAL));

  return handle;
}

void destroy_batchnorm_handle(BatchNormHandle *handle) {
  if (handle) {
    cudnnDestroyTensorDescriptor(handle->input_descriptor);
    cudnnDestroyTensorDescriptor(handle->output_descriptor);
    cudnnDestroyTensorDescriptor(handle->bn_scale_bias_mean_var_descriptor);
    delete handle;
  }
}

template <typename T>
void run_forward_training(BatchNormHandle *handle, const T *input, const T *bnScale,
                          const T *bnBias, T *output, T *running_mean, T *running_var, T *save_mean,
                          T *save_inv_var, double epsilon, double exponential_average_factor,
                          cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = 1.0;
  T beta = 0.0;

  CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
      handle->cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, handle->input_descriptor, input,
      handle->output_descriptor, output, handle->bn_scale_bias_mean_var_descriptor, bnScale, bnBias,
      exponential_average_factor, running_mean, running_var, epsilon, save_mean, save_inv_var));
}

template <typename T>
void run_forward_inference(BatchNormHandle *handle, const T *input, const T *bnScale,
                           const T *bnBias, const T *estimated_mean, const T *estimated_var,
                           T *output, double epsilon, cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alpha = 1.0;
  T beta = 0.0;

  CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
      handle->cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, handle->input_descriptor, input,
      handle->output_descriptor, output, handle->bn_scale_bias_mean_var_descriptor, bnScale, bnBias,
      estimated_mean, estimated_var, epsilon));
}

template <typename T>
void run_backward(BatchNormHandle *handle, const T *input, const T *grad_output, const T *bnScale,
                  T *grad_input, T *grad_bnScale, T *grad_bnBias, const T *save_mean,
                  const T *save_inv_var, double epsilon, cudaStream_t stream) {
  CHECK_CUDNN(cudnnSetStream(handle->cudnn_handle, stream));

  T alphaDataDiff = 1.0;
  T betaDataDiff = 0.0;
  T alphaParamDiff = 1.0;
  T betaParamDiff = 0.0;

  CHECK_CUDNN(cudnnBatchNormalizationBackward(
      handle->cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alphaDataDiff, &betaDataDiff, &alphaParamDiff,
      &betaParamDiff, handle->input_descriptor, input, handle->output_descriptor, grad_output,
      handle->input_descriptor, grad_input, handle->bn_scale_bias_mean_var_descriptor, bnScale,
      grad_bnScale, grad_bnBias, epsilon, save_mean, save_inv_var));
}

template void run_forward_training<float>(BatchNormHandle *handle, const float *input,
                                          const float *bnScale, const float *bnBias, float *output,
                                          float *running_mean, float *running_var, float *save_mean,
                                          float *save_inv_var, double epsilon,
                                          double exponential_average_factor, cudaStream_t stream);
template void run_forward_training<double>(BatchNormHandle *handle, const double *input,
                                           const double *bnScale, const double *bnBias,
                                           double *output, double *running_mean,
                                           double *running_var, double *save_mean,
                                           double *save_inv_var, double epsilon,
                                           double exponential_average_factor, cudaStream_t stream);

template void run_forward_inference<float>(BatchNormHandle *handle, const float *input,
                                           const float *bnScale, const float *bnBias,
                                           const float *estimated_mean, const float *estimated_var,
                                           float *output, double epsilon, cudaStream_t stream);
template void run_forward_inference<double>(BatchNormHandle *handle, const double *input,
                                            const double *bnScale, const double *bnBias,
                                            const double *estimated_mean,
                                            const double *estimated_var, double *output,
                                            double epsilon, cudaStream_t stream);

template void run_backward<float>(BatchNormHandle *handle, const float *input,
                                  const float *grad_output, const float *bnScale, float *grad_input,
                                  float *grad_bnScale, float *grad_bnBias, const float *save_mean,
                                  const float *save_inv_var, double epsilon, cudaStream_t stream);
template void run_backward<double>(BatchNormHandle *handle, const double *input,
                                   const double *grad_output, const double *bnScale,
                                   double *grad_input, double *grad_bnScale, double *grad_bnBias,
                                   const double *save_mean, const double *save_inv_var,
                                   double epsilon, cudaStream_t stream);

} // namespace cudnn_batchnorm
} // namespace cuda
} // namespace tnn

#endif
