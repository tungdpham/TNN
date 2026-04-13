/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

#include "nn/layers_impl/common/n_ary.hpp"
#include "type/type.hpp"
namespace tnn {
namespace cpu {
namespace nary {
template <typename T>
void run_forward(const Vec<const T *> &inputs, T *output, const Vec<size_t> &shape,
                 const NAryOp &op_type);

// Backward pass: computes gradients with respect to each input
template <typename T>
void run_backward(const T *grad_output, Vec<T *> &grad_inputs, const Vec<const T *> &fwd_inputs,
                  const Vec<size_t> &shape, const NAryOp &op_type);

}  // namespace nary
}  // namespace cpu
}  // namespace tnn
