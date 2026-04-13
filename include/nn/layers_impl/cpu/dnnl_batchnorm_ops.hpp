/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_DNNL

#include <dnnl.hpp>

#include "nn/layers_impl/common/batchnorm.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dnnl_batchnorm {
struct dnnlBNHandle_t;

dnnlBNHandle_t *initialize_dnnl_handle(BatchNormStats &stats, DType_t dtype);

void destroy_dnnl_handle(dnnlBNHandle_t *handle);

void run_forward(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                 const void *scale_data, const void *shift_data, void *output_data, void *mean_data,
                 void *var_data, void *relu_ws_data, void *scratchpad_data);

void run_inference(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                   const void *scale_data, const void *shift_data, const void *mean_data,
                   const void *var_data, void *output_data, void *scratchpad_data);

void run_backward(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                  const void *grad_output_data, void *grad_input_data, const void *scale_data,
                  void *d_scale_data, void *d_shift_data, const void *mean_data,
                  const void *var_data, const void *relu_ws_data, void *scratchpad_data);

}  // namespace dnnl_batchnorm
}  // namespace cpu
}  // namespace tnn

#endif  // USE_DNNL
