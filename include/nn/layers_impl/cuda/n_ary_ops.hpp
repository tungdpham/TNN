/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <vector>

#include "nn/layers_impl/common/n_ary.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tnn {

namespace cuda {

// workspace must be a device pointer of at least nary_forward_workspace_bytes(n_inputs) bytes
template <typename T>
void nary_forward(const std::vector<const T *> &inputs, T *output, const std::vector<size_t> &shape,
                  const NAryOp &op_type, void *workspace, cudaStream_t stream);

// workspace must be a device pointer of at least nary_backward_workspace_bytes(n_inputs) bytes
template <typename T>
void nary_backward(const T *grad_output, std::vector<T *> &grad_inputs,
                   const std::vector<const T *> &fwd_inputs, const std::vector<size_t> &shape,
                   const NAryOp &op_type, void *workspace, cudaStream_t stream);

// Returns the number of bytes needed in the workspace for n_inputs inputs.
inline size_t nary_forward_workspace_bytes(size_t n_inputs) { return n_inputs * sizeof(void *); }

inline size_t nary_backward_workspace_bytes(size_t n_inputs) {
  return 2 * n_inputs * sizeof(void *);
}

}  // namespace cuda
}  // namespace tnn
