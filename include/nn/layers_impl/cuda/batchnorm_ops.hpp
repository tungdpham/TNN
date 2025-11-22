#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace batchnorm {

// --------------------------------------------------------------------------
// Optimized Fused Functions (New - Recommended for best performance)
// --------------------------------------------------------------------------
template <typename T>
void run_forward_fused(const T *input, T *mean, T *inv_std, T *running_mean, T *running_var,
                       const T *gamma, const T *beta, T *output, T *norm_cache, size_t N, size_t C,
                       size_t S, T momentum, T epsilon, bool affine, cudaStream_t stream);

template <typename T>
void run_backward_fused(const T *grad_output, const T *norm_input, const T *inv_std, const T *gamma,
                        T *d_gamma, T *d_beta, T *grad_input, size_t N, size_t C, size_t S,
                        bool affine, cudaStream_t stream);

// --------------------------------------------------------------------------
// Legacy Functions (Kept for backward compatibility)
// --------------------------------------------------------------------------
template <typename T>
void compute_mean_variance_fused(const T *input_data, T *mean_data, T *var_data, size_t batch_size,
                                 size_t channels, size_t spatial_size, cudaStream_t stream);

template <typename T>
void normalize_and_scale(const T *input_data, const T *mean_data, const T *std_data,
                         const T *gamma_data, const T *beta_data, T *output_data,
                         T *normalized_data, size_t batch_size, size_t channels,
                         size_t spatial_size, bool affine, cudaStream_t stream);

template <typename T>
void compute_batch_std(const T *batch_var_data, T *batch_std_data, size_t channels, T epsilon,
                       cudaStream_t stream);

template <typename T>
void update_running_stats(T *running_mean_data, T *running_var_data, const T *batch_mean_data,
                          const T *batch_var_data, size_t channels, T momentum,
                          cudaStream_t stream);

template <typename T>
void compute_inference_output(const T *input_data, const T *running_mean_data,
                              const T *running_var_data, const T *gamma_data, const T *beta_data,
                              T *output_data, size_t batch_size, size_t channels,
                              size_t spatial_size, T epsilon, bool affine, cudaStream_t stream);

template <typename T>
void compute_batchnorm_backward_fused(const T *gradient_data, const T *normalized_data,
                                      const T *std_data, const T *gamma_data, T *grad_input_data,
                                      T *gamma_grad, T *beta_grad, size_t batch_size,
                                      size_t channels, size_t spatial_size, bool affine,
                                      cudaStream_t stream,
                                      T *workspace_sum_grad_normalized = nullptr,
                                      T *workspace_sum_grad_norm_times_norm = nullptr);

} // namespace batchnorm
} // namespace cuda
} // namespace tnn