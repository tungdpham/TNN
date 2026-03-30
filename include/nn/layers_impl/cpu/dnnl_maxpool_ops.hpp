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

// Opaque handle caching compiled DNNL max-pooling primitives for a given shape to prevent
// recompilation on every call.
struct dnnlMaxPoolHandle_t;

// Build and cache DNNL primitives. Populates workspace sizes inside `stats`.
dnnlMaxPoolHandle_t *initialize_dnnl_handle(MaxPoolStats &stats, DType_t dtype);

void destroy_dnnl_handle(dnnlMaxPoolHandle_t *handle);

// Training forward: writes the DNNL pool workspace (index buffer) required by the backward pass.
// pool_workspace_data must point to at least stats.pool_workspace_size bytes.
// scratchpad_data may be nullptr when stats.fwd_workspace_size == 0.
void run_forward(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats, const void *input_data,
                 void *output_data, void *pool_workspace_data, void *scratchpad_data);

// Inference forward: no pool workspace produced or consumed.
// scratchpad_data may be nullptr when stats.inf_workspace_size == 0.
void run_forward_inference(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats,
                           const void *input_data, void *output_data, void *scratchpad_data);

// Backward: reads the DNNL pool workspace produced by the corresponding run_forward call.
// pool_workspace_data must be the same buffer that was filled by run_forward.
// scratchpad_data may be nullptr when stats.bwd_workspace_size == 0.
void run_backward(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats,
                  const void *grad_output_data, void *grad_input_data,
                  const void *pool_workspace_data, void *scratchpad_data);

}  // namespace dnnl_maxpool
}  // namespace cpu
}  // namespace tnn

#endif  // USE_DNNL
