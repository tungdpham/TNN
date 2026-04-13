/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_DNNL

#include <dnnl.hpp>

#include "nn/layers_impl/common/maxpool.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dnnl_maxpool {

struct dnnlMaxPoolHandle_t;

dnnlMaxPoolHandle_t *initialize_dnnl_handle(MaxPoolStats &stats, DType_t dtype);

void destroy_dnnl_handle(dnnlMaxPoolHandle_t *handle);

void run_forward(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats, const void *input_data,
                 void *output_data, void *pool_workspace_data, void *scratchpad_data);

void run_inference(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats, const void *input_data,
                   void *output_data, void *scratchpad_data);

void run_backward(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats,
                  const void *grad_output_data, void *grad_input_data,
                  const void *pool_workspace_data, void *scratchpad_data);

}  // namespace dnnl_maxpool
}  // namespace cpu
}  // namespace tnn

#endif  // USE_DNNL
