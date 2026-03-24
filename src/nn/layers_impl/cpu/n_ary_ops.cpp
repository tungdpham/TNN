/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/n_ary_ops.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "type/type.hpp"

namespace tnn {
namespace cpu {

// Helper to compute total number of elements
static size_t compute_total_elements(const Vec<size_t> &shape) {
  size_t total = 1;
  for (auto dim : shape) {
    total *= dim;
  }
  return total;
}

template <typename T>
void nary_forward(const Vec<const T *> &inputs, T *output, const Vec<size_t> &shape,
                  const NAryOp &op_type) {
  if (inputs.empty()) {
    throw std::runtime_error("nary_forward: requires at least one input");
  }

  size_t total_elements = compute_total_elements(shape);

  switch (op_type) {
    case NAryOp::ADD: {
      // Initialize output with first input
      for (size_t i = 0; i < total_elements; ++i) {
        output[i] = inputs[0][i];
      }
      // Add remaining inputs
      for (size_t input_idx = 1; input_idx < inputs.size(); ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          output[i] += inputs[input_idx][i];
        }
      }
      break;
    }
    case NAryOp::SUB: {
      // Initialize output with first input
      for (size_t i = 0; i < total_elements; ++i) {
        output[i] = inputs[0][i];
      }
      // Subtract remaining inputs
      for (size_t input_idx = 1; input_idx < inputs.size(); ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          output[i] -= inputs[input_idx][i];
        }
      }
      break;
    }
    case NAryOp::MUL: {
      // Initialize output with first input
      for (size_t i = 0; i < total_elements; ++i) {
        output[i] = inputs[0][i];
      }
      // Multiply by remaining inputs
      for (size_t input_idx = 1; input_idx < inputs.size(); ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          output[i] *= inputs[input_idx][i];
        }
      }
      break;
    }
    case NAryOp::DIV: {
      // Initialize output with first input
      for (size_t i = 0; i < total_elements; ++i) {
        output[i] = inputs[0][i];
      }
      // Divide by remaining inputs
      for (size_t input_idx = 1; input_idx < inputs.size(); ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          // Add small epsilon to avoid division by zero
          if (std::abs(static_cast<float>(inputs[input_idx][i])) < 1e-8) {
            throw std::runtime_error("nary_forward (DIV): division by near-zero value");
          }
          output[i] /= inputs[input_idx][i];
        }
      }
      break;
    }
    default:
      throw std::runtime_error("nary_forward: unknown operation type");
  }
}

template <typename T>
void nary_backward(const T *grad_output, Vec<T *> &grad_inputs, const Vec<const T *> &fwd_inputs,
                   const Vec<size_t> &shape, const NAryOp &op_type) {
  if (fwd_inputs.empty() || grad_inputs.empty()) {
    throw std::runtime_error("nary_backward: requires at least one input");
  }

  size_t total_elements = compute_total_elements(shape);
  size_t n_inputs = fwd_inputs.size();

  switch (op_type) {
    case NAryOp::ADD: {
      // For addition: d(output)/d(input_i) = 1
      // So: grad_input[i] = grad_output
      for (size_t input_idx = 0; input_idx < n_inputs; ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          grad_inputs[input_idx][i] += grad_output[i];
        }
      }
      break;
    }
    case NAryOp::SUB: {
      // For subtraction: output = i0 - i1 - i2 - ...
      // d(output)/d(i0) = 1, d(output)/d(i_k) = -1 for k > 0
      for (size_t i = 0; i < total_elements; ++i) {
        grad_inputs[0][i] += grad_output[i];
      }
      for (size_t input_idx = 1; input_idx < n_inputs; ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          grad_inputs[input_idx][i] -= grad_output[i];
        }
      }
      break;
    }
    case NAryOp::MUL: {
      // For multiplication: output = i0 * i1 * i2 * ...
      // d(output)/d(i_j) = output / i_j = product of all other inputs
      for (size_t input_idx = 0; input_idx < n_inputs; ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          T grad_contrib = grad_output[i];
          // Multiply by all inputs except input_idx
          for (size_t j = 0; j < n_inputs; ++j) {
            if (j != input_idx) {
              grad_contrib *= fwd_inputs[j][i];
            }
          }
          grad_inputs[input_idx][i] += grad_contrib;
        }
      }
      break;
    }
    case NAryOp::DIV: {
      // For division: output = ((i0 / i1) / i2) / ...
      // More complex chain rule computation
      // We need: d(output)/d(i_j)
      //
      // output = i0 / i1 / i2 / ... / in
      // = i0 / (i1 * i2 * ... * in)
      //
      // d(output)/d(i0) = 1 / (i1 * i2 * ... * in)
      // d(output)/d(i_j) = -i0 / (i1 * i2 * ... * in * i_j) for j >= 1

      // Compute denominator product: i1 * i2 * ... * in
      Vec<T> denom_prod(total_elements, static_cast<T>(1.0));
      for (size_t input_idx = 1; input_idx < n_inputs; ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          denom_prod[i] *= fwd_inputs[input_idx][i];
        }
      }

      // Gradient for i0
      for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(static_cast<float>(denom_prod[i])) < 1e-8) {
          throw std::runtime_error("nary_backward (DIV): near-zero denominator");
        }
        grad_inputs[0][i] += grad_output[i] / denom_prod[i];
      }

      // Gradients for i_j (j >= 1)
      for (size_t input_idx = 1; input_idx < n_inputs; ++input_idx) {
        for (size_t i = 0; i < total_elements; ++i) {
          if (std::abs(static_cast<float>(fwd_inputs[input_idx][i])) < 1e-8) {
            throw std::runtime_error("nary_backward (DIV): near-zero divisor");
          }
          T grad_contrib =
              -grad_output[i] * fwd_inputs[0][i] / (denom_prod[i] * fwd_inputs[input_idx][i]);
          grad_inputs[input_idx][i] += grad_contrib;
        }
      }
      break;
    }
    default:
      throw std::runtime_error("nary_backward: unknown operation type");
  }
}

#define INSTANTIATE_NARY_OPS(T)                                                                    \
  template void nary_forward<T>(const Vec<const T *> &, T *, const Vec<size_t> &, const NAryOp &); \
  template void nary_backward<T>(const T *, Vec<T *> &, const Vec<const T *> &,                    \
                                 const Vec<size_t> &, const NAryOp &);

INSTANTIATE_NARY_OPS(fp16)
INSTANTIATE_NARY_OPS(bf16)
INSTANTIATE_NARY_OPS(float)
INSTANTIATE_NARY_OPS(double)

#undef INSTANTIATE_NARY_OPS

}  // namespace cpu
}  // namespace tnn
