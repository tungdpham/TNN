#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace batchnorm_nchw {

template <typename T>
void compute_inference_output(const T *input_data, const float *running_mean_data,
                              const float *running_var_data, const float *gamma_data,
                              const float *beta_data, T *output_data, size_t batch_size,
                              size_t channels, size_t spatial_size, float epsilon, bool affine);

template <typename T>
void run_forward_fused(const T *input, float *mean, float *inv_std, float *running_mean,
                       float *running_var, const float *gamma, const float *beta, T *output,
                       float *norm_cache, size_t N, size_t C, size_t S, float momentum,
                       float epsilon, bool affine);

template <typename T>
void run_backward_fused(const T *grad_output, const float *norm_input, const float *inv_std,
                        const float *gamma, float *d_gamma, float *d_beta, T *grad_input, size_t N,
                        size_t C, size_t S, bool affine);

} // namespace batchnorm_nchw
} // namespace cpu
} // namespace tnn