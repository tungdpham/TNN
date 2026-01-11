/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/cuda/softmax.hpp"

#ifdef USE_CUDNN
#include <iostream>
#include <stdexcept>

namespace tnn {
namespace cuda {

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

template <typename T>
void softmax_forward(cudnnHandle_t handle, const T *input, T *output, size_t rows, size_t cols,
                     cudaStream_t stream) {
  cudnnSetStream(handle, stream);
  cudnnTensorDescriptor_t desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, get_cudnn_data_type<T>(), rows,
                                         cols, 1, 1));

  T alpha = 1.0f;
  T beta = 0.0f;

  CHECK_CUDNN(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha, desc, input, &beta, desc, output));

  CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
}

template <typename T>
void softmax_backward(cudnnHandle_t handle, const T *output, const T *grad_output, T *grad_input,
                      size_t rows, size_t cols, cudaStream_t stream) {
  cudnnSetStream(handle, stream);
  cudnnTensorDescriptor_t desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, get_cudnn_data_type<T>(), rows,
                                         cols, 1, 1));

  T alpha = 1.0f;
  T beta = 0.0f;

  CHECK_CUDNN(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha, desc, output, desc, grad_output, &beta, desc,
                                   grad_input));

  CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
}

template void softmax_forward<float>(cudnnHandle_t handle, const float *input, float *output,
                                     size_t rows, size_t cols, cudaStream_t stream);

template void softmax_backward<float>(cudnnHandle_t handle, const float *output,
                                      const float *grad_output, float *grad_input, size_t rows,
                                      size_t cols, cudaStream_t stream);

template void softmax_forward<double>(cudnnHandle_t handle, const double *input, double *output,
                                      size_t rows, size_t cols, cudaStream_t stream);

template void softmax_backward<double>(cudnnHandle_t handle, const double *output,
                                       const double *grad_output, double *grad_input, size_t rows,
                                       size_t cols, cudaStream_t stream);

} // namespace cuda
} // namespace tnn
#endif
