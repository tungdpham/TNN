/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_DNNL

#include <dnnl.hpp>

#include "nn/layers_impl/common/conv2d.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dnnl_conv2d {

// Opaque handle that caches compiled DNNL convolution primitives for a given shape to prevent
// recompilation on every call.
struct dnnlHandle_t;

dnnl::memory::data_type get_dnnl_dtype(DType_t dtype);

// Build and cache DNNL convolution primitives. Populates workspace sizes inside `stats`.
dnnlHandle_t *initialize_dnnl_handle(ConvolutionStats &stats, DType_t dtype);
void destroy_dnnl_handle(dnnlHandle_t *handle);

void run_forward(dnnlHandle_t *handle, const ConvolutionStats &stats, const void *input_data,
                 const void *weight_data, const void *bias_data, void *output_data,
                 void *workspace_data);

void run_backward_data(dnnlHandle_t *handle, const ConvolutionStats &stats,
                       const void *grad_output_data, const void *weight_data, void *grad_input_data,
                       void *workspace_data);

// Computes weight gradients and (if use_bias) bias gradients in one primitive execution.
void run_backward_weights_and_bias(dnnlHandle_t *handle, const ConvolutionStats &stats,
                                   const void *input_data, const void *grad_output_data,
                                   void *grad_weight_data, void *grad_bias_data,
                                   void *workspace_data);

}  // namespace dnnl_conv2d
}  // namespace cpu
}  // namespace tnn

#endif  // USE_DNNL