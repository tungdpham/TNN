/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>

#include "nn/layers_impl/common/layer_norm.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_layer_norm {

struct feHandle_t;

cudnnDataType_t get_cudnn_data_type(DType_t dtype);

feHandle_t *initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, LayerNormStats &stats);

void destroy_fe_handle(feHandle_t *handle);

void run_forward(feHandle_t *handle, const LayerNormStats &stats, const void *input_data,
                 const void *gamma_data, const void *beta_data, void *output_data, void *mean_data,
                 void *inv_variance_data, void *workspace_data, cudaStream_t stream);

void run_backward(feHandle_t *handle, const LayerNormStats &stats, const void *gradient_data,
                  const void *input_data, const void *gamma_data, const void *mean_data,
                  const void *inv_variance_data, void *grad_input_data, void *gamma_grad_data,
                  void *beta_grad_data, void *workspace_data, cudaStream_t stream);

}  // namespace cudnn_layer_norm
}  // namespace cuda
}  // namespace tnn

#endif
