/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/batchnorm_ops.hpp"

#include "ops/cpu/kernels.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {
namespace batchnorm {
template <typename T>
void compute_inference_output(const T *input_data, const T *running_mean_data,
                              const T *running_var_data, const T *gamma_data, const T *beta_data,
                              T *output_data, size_t batch_size, size_t channels,
                              size_t spatial_size, T epsilon, bool affine) {
  const size_t channel_stride = channels * spatial_size;

  parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    T mean_val = running_mean_data[c];
    T var_val = running_var_data[c];
    T std_val = std::sqrt(var_val + epsilon);
    const T inv_std = T(1) / std_val;

    const size_t base_idx = n * channel_stride + c * spatial_size;

    const T *input_ptr = input_data + base_idx;
    T *output_ptr = output_data + base_idx;

    if (affine) {
      const T gamma_val = gamma_data[c];
      const T beta_val = beta_data[c];

      for (size_t i = 0; i < spatial_size; ++i) {
        T normalized_val = (input_ptr[i] - mean_val) * inv_std;
        output_ptr[i] = gamma_val * normalized_val + beta_val;
      }
    } else {
      for (size_t i = 0; i < spatial_size; ++i) {
        output_ptr[i] = (input_ptr[i] - mean_val) * inv_std;
      }
    }
  });
}

// Fused forward matching GPU behavior: compute mean, inv_std, update running stats, normalize
template <typename T>
void run_forward_fused(const T *input, T *mean, T *inv_std, T *running_mean, T *running_var,
                       const T *gamma, const T *beta, T *output, T *norm_cache, size_t N, size_t C,
                       size_t S, T momentum, T epsilon, bool affine) {
  const size_t total_elements = N * S;
  const size_t channel_stride = C * S;
  const T inv_total = T(1) / static_cast<T>(total_elements);

  // Pass 1: Compute mean, variance, inv_std, and update running stats (fused)
  parallel_for<size_t>(0, C, [&](size_t c) {
    T sum = T(0);
    T sq_sum = T(0);
    const size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      const size_t n_offset = n * channel_stride;
      const size_t base_idx = n_offset + c_offset;
      const T *input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        T val = input_ptr[s];
        sum += val;
        sq_sum += val * val;
      }
    }

    T mu = sum * inv_total;
    T var = (sq_sum * inv_total) - (mu * mu);

    // Store mean and inv_std (matching GPU behavior)
    mean[c] = mu;
    inv_std[c] = T(1) / std::sqrt(var + epsilon);

    // Update running stats (fused to avoid separate pass)
    running_mean[c] = (T(1) - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (T(1) - momentum) * running_var[c] + momentum * var;
  });

  // Pass 2: Apply normalization using inv_std
  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const T mu = mean[c];
    const T istd = inv_std[c];

    const size_t n_offset = n * channel_stride;
    const size_t c_offset = c * S;
    const size_t base_idx = n_offset + c_offset;

    const T *input_ptr = input + base_idx;
    T *output_ptr = output + base_idx;
    T *norm_ptr = norm_cache ? (norm_cache + base_idx) : nullptr;

    for (size_t s = 0; s < S; ++s) {
      T x = input_ptr[s];
      T norm = (x - mu) * istd;

      if (norm_ptr)
        norm_ptr[s] = norm;

      if (affine) {
        output_ptr[s] = norm * gamma[c] + beta[c];
      } else {
        output_ptr[s] = norm;
      }
    }
  });
}

// Fused backward matching GPU behavior: use inv_std, compute gradients efficiently
template <typename T>
void run_backward_fused(const T *grad_output, const T *norm_input, const T *inv_std, const T *gamma,
                        T *d_gamma, T *d_beta, T *grad_input, size_t N, size_t C, size_t S,
                        bool affine) {
  const size_t channel_stride = C * S;
  const size_t M = N * S;
  const T inv_M = T(1) / static_cast<T>(M);

  // Pass 1: Compute gradient sums per channel
  parallel_for<size_t>(0, C, [&](size_t c) {
    T sum_dy = T(0);
    T sum_dy_x_norm = T(0);
    const size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      const size_t n_offset = n * channel_stride;
      const size_t base_idx = n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        size_t idx = base_idx + s;
        T dy = grad_output[idx];
        T x_hat = norm_input[idx];

        sum_dy += dy;
        sum_dy_x_norm += dy * x_hat;
      }
    }

    d_gamma[c] = sum_dy_x_norm;
    d_beta[c] = sum_dy;
  });

  // Pass 2: Compute input gradients using the formula matching GPU
  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const T g = affine ? gamma[c] : T(1);
    const T istd = inv_std[c];
    const T sum_dy = d_beta[c];
    const T sum_dy_x_norm = d_gamma[c];

    const size_t n_offset = n * channel_stride;
    const size_t c_offset = c * S;
    const size_t base_idx = n_offset + c_offset;

    const T term1 = (g * istd) * inv_M;

    for (size_t s = 0; s < S; ++s) {
      size_t idx = base_idx + s;
      T dy = grad_output[idx];
      T x_hat = norm_input[idx];

      // Standard BN Backward formula: dx = (1/M) * (gamma / std) * (M * dy - sum_dy - x_hat *
      // sum_dy_x_norm)
      T term2 = static_cast<T>(M) * dy - sum_dy - (x_hat * sum_dy_x_norm);
      grad_input[idx] = term1 * term2;
    }
  });
}

template void
compute_inference_output<float>(const float *input_data, const float *running_mean_data,
                                const float *running_var_data, const float *gamma_data,
                                const float *beta_data, float *output_data, size_t batch_size,
                                size_t channels, size_t spatial_size, float epsilon, bool affine);
template void
compute_inference_output<double>(const double *input_data, const double *running_mean_data,
                                 const double *running_var_data, const double *gamma_data,
                                 const double *beta_data, double *output_data, size_t batch_size,
                                 size_t channels, size_t spatial_size, double epsilon, bool affine);

template void run_forward_fused<float>(const float *input, float *mean, float *inv_std,
                                       float *running_mean, float *running_var, const float *gamma,
                                       const float *beta, float *output, float *norm_cache,
                                       size_t N, size_t C, size_t S, float momentum, float epsilon,
                                       bool affine);
template void run_forward_fused<double>(const double *input, double *mean, double *inv_std,
                                        double *running_mean, double *running_var,
                                        const double *gamma, const double *beta, double *output,
                                        double *norm_cache, size_t N, size_t C, size_t S,
                                        double momentum, double epsilon, bool affine);

template void run_backward_fused<float>(const float *grad_output, const float *norm_input,
                                        const float *inv_std, const float *gamma, float *d_gamma,
                                        float *d_beta, float *grad_input, size_t N, size_t C,
                                        size_t S, bool affine);
template void run_backward_fused<double>(const double *grad_output, const double *norm_input,
                                         const double *inv_std, const double *gamma,
                                         double *d_gamma, double *d_beta, double *grad_input,
                                         size_t N, size_t C, size_t S, bool affine);

} // namespace batchnorm
} // namespace cpu
} // namespace tnn
