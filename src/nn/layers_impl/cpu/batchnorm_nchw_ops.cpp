/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/batchnorm_nchw_ops.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace batchnorm_nchw {
template <typename T>
void compute_inference_output(const T *input_data, const float *running_mean_data,
                              const float *running_var_data, const float *gamma_data,
                              const float *beta_data, T *output_data, size_t batch_size,
                              size_t channels, size_t spatial_size, float epsilon, bool affine) {
  const size_t channel_stride = channels * spatial_size;

  parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    float mean_val = running_mean_data[c];
    float var_val = running_var_data[c];
    float std_val = std::sqrt(var_val + epsilon);
    const float inv_std = 1.0f / std_val;

    const size_t base_idx = n * channel_stride + c * spatial_size;

    const T *input_ptr = input_data + base_idx;
    T *output_ptr = output_data + base_idx;

    if (affine) {
      const float gamma_val = gamma_data[c];
      const float beta_val = beta_data[c];

      for (size_t i = 0; i < spatial_size; ++i) {
        float normalized_val = (static_cast<float>(input_ptr[i]) - mean_val) * inv_std;
        output_ptr[i] = static_cast<T>(gamma_val * normalized_val + beta_val);
      }
    } else {
      for (size_t i = 0; i < spatial_size; ++i) {
        output_ptr[i] = static_cast<T>((static_cast<float>(input_ptr[i]) - mean_val) * inv_std);
      }
    }
  });
}

template <typename T>
void run_forward_fused(const T *input, float *mean, float *inv_std, float *running_mean,
                       float *running_var, const float *gamma, const float *beta, T *output,
                       float *norm_cache, size_t N, size_t C, size_t S, float momentum,
                       float epsilon, bool affine) {
  const size_t total_elements = N * S;
  const size_t channel_stride = C * S;
  const float inv_total = 1.0f / static_cast<float>(total_elements);

  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum = 0.0f;
    const size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      const size_t n_offset = n * channel_stride;
      const size_t base_idx = n_offset + c_offset;
      const T *input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        sum += static_cast<float>(input_ptr[s]);
      }
    }

    float mu = sum * inv_total;
    mean[c] = mu;

    float var_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      const size_t n_offset = n * channel_stride;
      const size_t base_idx = n_offset + c_offset;
      const T *input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        float diff = static_cast<float>(input_ptr[s]) - mu;
        var_sum += diff * diff;
      }
    }

    float var = var_sum * inv_total;

    inv_std[c] = 1.0f / std::sqrt(var + epsilon);

    float unbiased_var = var_sum / static_cast<float>(total_elements - 1);

    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  });

  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const float mu = mean[c];
    const float istd = inv_std[c];

    const size_t n_offset = n * channel_stride;
    const size_t c_offset = c * S;
    const size_t base_idx = n_offset + c_offset;

    const T *input_ptr = input + base_idx;
    T *output_ptr = output + base_idx;
    float *norm_ptr = norm_cache ? (norm_cache + base_idx) : nullptr;

    for (size_t s = 0; s < S; ++s) {
      float x = static_cast<float>(input_ptr[s]);
      float norm = (x - mu) * istd;

      if (norm_ptr) norm_ptr[s] = norm;

      if (affine) {
        output_ptr[s] = static_cast<T>(norm * gamma[c] + beta[c]);
      } else {
        output_ptr[s] = static_cast<T>(norm);
      }
    }
  });
}

template <typename T>
void run_backward_fused(const T *grad_output, const float *norm_input, const float *inv_std,
                        const float *gamma, float *d_gamma, float *d_beta, T *grad_input, size_t N,
                        size_t C, size_t S, bool affine) {
  const size_t channel_stride = C * S;
  const size_t M = N * S;
  const float inv_M = 1.0f / static_cast<float>(M);

  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum_dy = 0.0f;
    float sum_dy_x_norm = 0.0f;
    const size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      const size_t n_offset = n * channel_stride;
      const size_t base_idx = n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        size_t idx = base_idx + s;
        float dy = static_cast<float>(grad_output[idx]);
        float x_hat = norm_input[idx];

        sum_dy += dy;
        sum_dy_x_norm += dy * x_hat;
      }
    }

    if (affine) {
      d_gamma[c] += sum_dy_x_norm;
      d_beta[c] += sum_dy;
    } else {
      d_gamma[c] = sum_dy_x_norm;
      d_beta[c] = sum_dy;
    }
  });

  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const float g = (affine && gamma) ? gamma[c] : 1.0f;
    const float istd = inv_std[c];

    const float sum_dy = d_beta[c];
    const float sum_dy_x_norm = d_gamma[c];

    const size_t n_offset = n * channel_stride;
    const size_t c_offset = c * S;
    const size_t base_idx = n_offset + c_offset;

    const float term1 = (g * istd) * inv_M;

    for (size_t s = 0; s < S; ++s) {
      size_t idx = base_idx + s;
      float dy = static_cast<float>(grad_output[idx]);
      float x_hat = norm_input[idx];

      float term2 = static_cast<float>(M) * dy - sum_dy - (x_hat * sum_dy_x_norm);
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

#define INSTANTIATE_BATCHNORM(T)                                                               \
  template void compute_inference_output<T>(                                                   \
      const T *input_data, const float *running_mean_data, const float *running_var_data,      \
      const float *gamma_data, const float *beta_data, T *output_data, size_t batch_size,      \
      size_t channels, size_t spatial_size, float epsilon, bool affine);                       \
                                                                                               \
  template void run_forward_fused<T>(                                                          \
      const T *input, float *mean, float *inv_std, float *running_mean, float *running_var,    \
      const float *gamma, const float *beta, T *output, float *norm_cache, size_t N, size_t C, \
      size_t S, float momentum, float epsilon, bool affine);                                   \
                                                                                               \
  template void run_backward_fused<T>(                                                         \
      const T *grad_output, const float *norm_input, const float *inv_std, const float *gamma, \
      float *d_gamma, float *d_beta, T *grad_input, size_t N, size_t C, size_t S, bool affine);
INSTANTIATE_BATCHNORM(fp16)
INSTANTIATE_BATCHNORM(bf16)
INSTANTIATE_BATCHNORM(float)
INSTANTIATE_BATCHNORM(double)
#undef INSTANTIATE_BATCHNORM

}  // namespace batchnorm_nchw
}  // namespace cpu
}  // namespace tnn
