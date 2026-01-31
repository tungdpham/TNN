#pragma once

#ifdef USE_CUDNN
#include <cuda_runtime.h>
#include <cudnn.h>

#include "math/common/gemm.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_gemm {

struct feHandle_t;

feHandle_t *initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_type,
                                 cudnnDataType_t param_type, cudnnDataType_t compute_type,
                                 GemmStats &stats);

void destroy_fe_handle(feHandle_t *handle);

void run_forward(feHandle_t *handle, const GemmStats &stats, const void *input_data,
                 const void *weight_data, void *output_data, void *workspace_data,
                 cudaStream_t stream);

void run_dgrad(feHandle_t *handle, const GemmStats &stats, const void *gradient_data,
               const void *weight_data, void *grad_input_data, void *workspace_data,
               cudaStream_t stream);

void run_wgrad(feHandle_t *handle, const GemmStats &stats, const void *input_data,
               const void *gradient_data, void *weight_grad_data, void *workspace_data,
               cudaStream_t stream);

}  // namespace cudnn_gemm
}  // namespace cuda
}  // namespace tnn

#endif
