#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace groupnorm {

template <typename T>
void run_forward_fused(const T *input, T *mean, T *inv_std, const T *gamma, const T *beta,
                       T *output, T *norm_cache, size_t N, size_t C, size_t S, size_t num_groups,
                       T epsilon, bool affine);

template <typename T>
void run_backward_fused(const T *grad_output, const T *norm_input, const T *inv_std, const T *gamma,
                        T *d_gamma, T *d_beta, T *grad_input, size_t N, size_t C, size_t S,
                        size_t num_groups, bool affine);

} // namespace groupnorm
} // namespace cpu
} // namespace tnn
