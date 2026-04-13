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
template <typename T>
void run_inference(const T *input, const float *running_mean, const float *running_var,
                   const float *gamma, const float *beta, T *output, size_t N, size_t C, size_t S,
                   float epsilon, bool affine);
template <typename T>
void run_forward(const T *input, float *mean, float *inv_std, float *running_mean,
                 float *running_var, const float *gamma, const float *beta, T *output,
                 bool *relu_mask, size_t N, size_t C, size_t S, float momentum, float epsilon,
                 bool affine, bool use_relu);
template <typename T>
void run_backward(const T *grad_output, const T *input, const float *mean, const float *inv_std,
                  const float *gamma, float *d_gamma, float *d_beta, T *grad_input,
                  const bool *relu_mask, size_t N, size_t C, size_t S, bool affine, bool use_relu);

}  // namespace batchnorm_nhwc
}  // namespace cpu
}  // namespace tnn
