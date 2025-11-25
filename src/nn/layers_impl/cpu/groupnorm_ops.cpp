/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/groupnorm_ops.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
namespace groupnorm {

// Fused forward: compute mean, inv_std per group, normalize
template <typename T>
void run_forward_fused(const T *input, T *mean, T *inv_std, const T *gamma, const T *beta,
                       T *output, T *norm_cache, size_t N, size_t C, size_t S, size_t num_groups,
                       T epsilon, bool affine) {
  const size_t channels_per_group = C / num_groups;
  const size_t group_size = channels_per_group * S;
  const size_t channel_stride = C * S;
  const T inv_group_size = T(1) / static_cast<T>(group_size);

  // Pass 1: Compute mean and inv_std per group
  parallel_for_2d(N, num_groups, [&](size_t n, size_t g) {
    T sum = T(0);
    const size_t n_offset = n * channel_stride;

    // First pass: compute mean for this group
    for (size_t c = 0; c < channels_per_group; ++c) {
      const size_t c_offset = (g * channels_per_group + c) * S;
      const T *input_ptr = input + n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        sum += input_ptr[s];
      }
    }

    T mu = sum * inv_group_size;
    const size_t group_idx = n * num_groups + g;
    mean[group_idx] = mu;

    // Second pass: compute variance
    T var_sum = T(0);
    for (size_t c = 0; c < channels_per_group; ++c) {
      const size_t c_offset = (g * channels_per_group + c) * S;
      const T *input_ptr = input + n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        T diff = input_ptr[s] - mu;
        var_sum += diff * diff;
      }
    }

    T var = var_sum * inv_group_size;
    inv_std[group_idx] = T(1) / std::sqrt(var + epsilon);
  });

  // Pass 2: Apply normalization
  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const size_t g = c / channels_per_group;
    const size_t group_idx = n * num_groups + g;
    const T mu = mean[group_idx];
    const T istd = inv_std[group_idx];

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

// Fused backward: compute gradients using inv_std
template <typename T>
void run_backward_fused(const T *grad_output, const T *norm_input, const T *inv_std, const T *gamma,
                        T *d_gamma, T *d_beta, T *grad_input, size_t N, size_t C, size_t S,
                        size_t num_groups, bool affine) {
  const size_t channels_per_group = C / num_groups;
  const size_t group_size = channels_per_group * S;
  const size_t channel_stride = C * S;
  const T inv_group_size = T(1) / static_cast<T>(group_size);

  if (affine) {
    // Pass 1: Compute parameter gradients per channel
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

      d_gamma[c] += sum_dy_x_norm;
      d_beta[c] += sum_dy;
    });
  }

  // Pass 2: Compute input gradients per group
  parallel_for_2d(N, num_groups, [&](size_t n, size_t g) {
    const size_t group_idx = n * num_groups + g;
    const T istd = inv_std[group_idx];
    const size_t n_offset = n * channel_stride;

    // Compute sums over the group
    T sum_dy = T(0);
    T sum_dy_x_norm = T(0);

    for (size_t c = 0; c < channels_per_group; ++c) {
      const size_t global_c = g * channels_per_group + c;
      const size_t c_offset = global_c * S;
      const size_t base_idx = n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        size_t idx = base_idx + s;
        T dy = grad_output[idx];
        T x_hat = norm_input[idx];
        T gamma_val = (affine && gamma) ? gamma[global_c] : T(1);

        sum_dy += dy * gamma_val;
        sum_dy_x_norm += dy * gamma_val * x_hat;
      }
    }

    // Compute gradients for each element in the group
    for (size_t c = 0; c < channels_per_group; ++c) {
      const size_t global_c = g * channels_per_group + c;
      const T gamma_val = (affine && gamma) ? gamma[global_c] : T(1);
      const size_t c_offset = global_c * S;
      const size_t base_idx = n_offset + c_offset;

      const T term1 = (gamma_val * istd) * inv_group_size;

      for (size_t s = 0; s < S; ++s) {
        size_t idx = base_idx + s;
        T dy = grad_output[idx];
        T x_hat = norm_input[idx];

        // GroupNorm backward formula: dx = (1/M) * (gamma / std) * (M * dy - sum_dy - x_hat *
        // sum_dy_x_norm)
        T term2 = static_cast<T>(group_size) * dy * gamma_val - sum_dy - (x_hat * sum_dy_x_norm);
        grad_input[idx] = term1 * term2;
      }
    }
  });
}

template void run_forward_fused<float>(const float *input, float *mean, float *inv_std,
                                       const float *gamma, const float *beta, float *output,
                                       float *norm_cache, size_t N, size_t C, size_t S,
                                       size_t num_groups, float epsilon, bool affine);
template void run_forward_fused<double>(const double *input, double *mean, double *inv_std,
                                        const double *gamma, const double *beta, double *output,
                                        double *norm_cache, size_t N, size_t C, size_t S,
                                        size_t num_groups, double epsilon, bool affine);

template void run_backward_fused<float>(const float *grad_output, const float *norm_input,
                                        const float *inv_std, const float *gamma, float *d_gamma,
                                        float *d_beta, float *grad_input, size_t N, size_t C,
                                        size_t S, size_t num_groups, bool affine);
template void run_backward_fused<double>(const double *grad_output, const double *norm_input,
                                         const double *inv_std, const double *gamma,
                                         double *d_gamma, double *d_beta, double *grad_input,
                                         size_t N, size_t C, size_t S, size_t num_groups,
                                         bool affine);

} // namespace groupnorm
} // namespace cpu
} // namespace tnn
