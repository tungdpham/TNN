#pragma once

#ifdef USE_CUDNN
#include "nn/layers_impl/common/batchnorm.hpp"
#include "type/type.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>

namespace tnn {
namespace cuda {
namespace cudnn_batchnorm {

struct feHandle_t;

cudnnDataType_t get_cudnn_data_type(DType_t dtype);

feHandle_t *initialize_fe_handle(cudnnHandle_t shared_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, BatchNormStats &stats);

void destroy_fe_handle(feHandle_t *handle);

void run_forward_training(feHandle_t *handle, const BatchNormStats &stats, const void *input,
                          const void *gamma, const void *beta, void *output,
                          void *prev_running_mean, void *prev_running_var, void *next_running_mean,
                          void *next_running_var, void *batch_mean, void *batch_invar,
                          void *workspace, cudaStream_t stream);

void run_forward_inference(feHandle_t *handle, const BatchNormStats &stats, const void *input,
                           const void *gamma, const void *beta, void *saved_mean, void *saved_invar,
                           void *output, void *workspace, cudaStream_t stream);

void run_backward(feHandle_t *handle, const BatchNormStats &stats, const void *input,
                  const void *grad_output, const void *gamma, void *grad_input, void *grad_gamma,
                  void *grad_beta, const void *batch_mean, const void *batch_invar, void *workspace,
                  cudaStream_t stream);

} // namespace cudnn_batchnorm
} // namespace cuda
} // namespace tnn

#endif
