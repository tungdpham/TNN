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

// Opaque handle that caches compiled DNNL batch normalization primitives for a given shape.
struct dnnlBNHandle_t;

// Build and cache DNNL batch normalization primitives. Populates workspace sizes inside `stats`.
dnnlBNHandle_t *initialize_dnnl_handle(BatchNormStats &stats, DType_t dtype);

void destroy_dnnl_handle(dnnlBNHandle_t *handle);

// Training forward: normalizes input using batch statistics, updates running stats,
// saves mean/var for backward. If use_relu is set, applies ReLU and stores workspace.
// relu_ws_data must be non-null when stats.relu_workspace_size > 0.
void run_forward_training(dnnlBNHandle_t *handle, const BatchNormStats &stats,
                          const void *input_data, const void *scale_data, const void *shift_data,
                          void *output_data, void *mean_data, void *var_data, void *relu_ws_data,
                          void *scratchpad_data);

// Inference forward: normalizes input using provided running mean/var.
// relu_ws_data and mean_data/var_data are outputs (may be nullptr for inference).
void run_forward_inference(dnnlBNHandle_t *handle, const BatchNormStats &stats,
                           const void *input_data, const void *scale_data, const void *shift_data,
                           const void *mean_data, const void *var_data, void *output_data,
                           void *scratchpad_data);

// Backward: computes grad_input, and (if affine) d_scale and d_shift (written, not accumulated).
// relu_ws_data must match what was saved during run_forward_training when use_relu is true.
void run_backward(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                  const void *grad_output_data, void *grad_input_data, const void *scale_data,
                  void *d_scale_data, void *d_shift_data, const void *mean_data,
                  const void *var_data, const void *relu_ws_data, void *scratchpad_data);

}  // namespace dnnl_batchnorm
}  // namespace cpu
}  // namespace tnn

#endif  // USE_DNNL
