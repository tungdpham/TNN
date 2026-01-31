#pragma once

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>

#include "nn/layers_impl/common/conv2d.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_conv2d {

struct feHandle_t;

cudnnDataType_t get_cudnn_data_type(DType_t dtype);

feHandle_t *initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, ConvolutionStats &stats);

void destroy_fe_handle(feHandle_t *handle);

void run_forward(feHandle_t *handle, const ConvolutionStats &stats, const void *input_data,
                 const void *weight_data, const void *bias_data, void *output_data,
                 void *workspace_data, cudaStream_t stream);

void run_backward_data(feHandle_t *handle, const ConvolutionStats &stats, const void *gradient_data,
                       const void *weight_data, void *input_grad_data, void *workspace_data,
                       cudaStream_t stream);

void run_backward_weights_and_bias(feHandle_t *handle, const ConvolutionStats &stats,
                                   const void *input_data, const void *gradient_data,
                                   void *weight_grad_data, void *bias_grad_data,
                                   void *workspace_data, cudaStream_t stream);

}  // namespace cudnn_conv2d
}  // namespace cuda
}  // namespace tnn

#endif
