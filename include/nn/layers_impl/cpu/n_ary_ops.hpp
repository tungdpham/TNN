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

namespace tnn {

class Layer;

namespace cpu {

// Forward pass: applies N-ary operation across inputs
// output = input[0] op input[1] op input[2] ... op input[N-1]
// For division: output = ((input[0] / input[1]) / input[2]) / ...
template <typename T>
void nary_forward(const std::vector<const T *> &inputs, T *output, const std::vector<size_t> &shape,
                  const NAryOp &op_type);

// Backward pass: computes gradients with respect to each input
template <typename T>
void nary_backward(const T *grad_output, std::vector<T *> &grad_inputs,
                   const std::vector<const T *> &fwd_inputs, const std::vector<size_t> &shape,
                   const NAryOp &op_type);

}  // namespace cpu
}  // namespace tnn
