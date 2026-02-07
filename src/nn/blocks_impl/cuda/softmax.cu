#include "nn/blocks_impl/cuda/softmax.hpp"

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>

#include "type/type.hpp"

namespace tnn {
namespace cuda {

template <typename T>
cudnnDataType_t get_cudnn_data_type();

template <>
inline cudnnDataType_t get_cudnn_data_type<float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
inline cudnnDataType_t get_cudnn_data_type<double>() {
  return CUDNN_DATA_DOUBLE;
}

template <>
inline cudnnDataType_t get_cudnn_data_type<fp16>() {
  return CUDNN_DATA_HALF;
}

template <>
inline cudnnDataType_t get_cudnn_data_type<bf16>() {
  return CUDNN_DATA_BFLOAT16;
}

template <typename T>
void softmax_forward(cudnnHandle_t handle, const T* input, T* output, size_t rows, size_t cols,
                     cudaStream_t stream) {
  cudnnSetStream(handle, stream);

  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, get_cudnn_data_type<T>(),
                             static_cast<int>(rows), static_cast<int>(cols), 1, 1);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                      input, &beta, desc, output);
  cudnnDestroyTensorDescriptor(desc);
}

template <typename T>
void softmax_backward(cudnnHandle_t handle, const T* output, const T* grad_output, T* grad_input,
                      size_t rows, size_t cols, cudaStream_t stream) {
  cudnnSetStream(handle, stream);

  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, get_cudnn_data_type<T>(),
                             static_cast<int>(rows), static_cast<int>(cols), 1, 1);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                       output, desc, grad_output, &beta, desc, grad_input);
  cudnnDestroyTensorDescriptor(desc);
}

#define INSTANTIATE_SOFTMAX(T)                                                                   \
  template void softmax_forward<T>(cudnnHandle_t handle, const T* input, T* output, size_t rows, \
                                   size_t cols, cudaStream_t stream);                            \
  template void softmax_backward<T>(cudnnHandle_t handle, const T* output, const T* grad_output, \
                                    T* grad_input, size_t rows, size_t cols, cudaStream_t stream);
INSTANTIATE_SOFTMAX(fp16);
INSTANTIATE_SOFTMAX(bf16);
INSTANTIATE_SOFTMAX(float);
INSTANTIATE_SOFTMAX(double);
#undef INSTANTIATE_SOFTMAX

}  // namespace cuda
}  // namespace tnn

#endif