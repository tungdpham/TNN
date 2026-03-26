/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace batchnorm_nhwc {

// Inference forward: normalizes input using running statistics.
// Input/output layout: NHWC — element [n, h, w, c] at index n*S*C + s*C + c, where S = H*W.
template <typename T>
void compute_inference_output(const T *input, const float *running_mean, const float *running_var,
                              const float *gamma, const float *beta, T *output, size_t N, size_t C,
                              size_t S, float epsilon, bool affine);

// Training forward: computes per-channel mean/inv_std, updates running statistics, normalizes
// input, optionally applies affine transform, and optionally applies ReLU while saving a mask.
// relu_mask must point to N*S*C bools when use_relu is true, otherwise may be nullptr.
template <typename T>
void run_forward_fused(const T *input, float *mean, float *inv_std, float *running_mean,
                       float *running_var, const float *gamma, const float *beta, T *output,
                       bool *relu_mask, size_t N, size_t C, size_t S, float momentum,
                       float epsilon, bool affine, bool use_relu);

// Backward: accumulates d_gamma / d_beta and computes grad_input.
// Recomputes x_hat on the fly from input, mean, and inv_std — no norm_cache needed.
// relu_mask must be the mask saved during the forward pass when use_relu is true.
template <typename T>
void run_backward_fused(const T *grad_output, const T *input, const float *mean,
                        const float *inv_std, const float *gamma, float *d_gamma, float *d_beta,
                        T *grad_input, const bool *relu_mask, size_t N, size_t C, size_t S,
                        bool affine, bool use_relu);

}  // namespace batchnorm_nhwc
}  // namespace cpu
}  // namespace tnn
