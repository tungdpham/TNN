/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/batchnorm_nhwc_ops.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace batchnorm_nhwc {

template <typename T>
void compute_inference_output(const T *input, const float *running_mean, const float *running_var,
                              const float *gamma, const float *beta, T *output, size_t N, size_t C,
                              size_t S, float epsilon, bool affine) {
  std::vector<float> scale(C);
  std::vector<float> bias(C);

  for (size_t c = 0; c < C; ++c) {
    float inv_std = 1.0f / std::sqrt(running_var[c] + epsilon);
    float g = affine ? gamma[c] : 1.0f;
    scale[c] = g * inv_std;
    bias[c] = beta[c] - (running_mean[c] * scale[c]);
  }

  const size_t M = N * S;
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float val = static_cast<float>(input[idx]);
      output[idx] = static_cast<T>(val * scale[c] + bias[c]);
    }
  });
}

template <typename T>
void run_forward_fused(const T *input, float *mean, float *inv_std, float *running_mean,
                       float *running_var, const float *gamma, const float *beta, T *output,
                       bool *relu_mask, size_t N, size_t C, size_t S, float momentum, float epsilon,
                       bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);

  std::vector<float> partial_sums(N * C, 0.0f);
  std::vector<float> partial_sq_sums(N * C, 0.0f);

  parallel_for<size_t>(0, N, [&](size_t n) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C; ++c) {
        size_t idx = n * S * C + s * C + c;
        float val = static_cast<float>(input[idx]);
        partial_sums[n * C + c] += val;
        partial_sq_sums[n * C + c] += val * val;
      }
    }
  });

  std::vector<float> scale(C);
  std::vector<float> bias_term(C);

  for (size_t c = 0; c < C; ++c) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      sum += partial_sums[n * C + c];
      sq_sum += partial_sq_sums[n * C + c];
    }

    float mu = sum * inv_M;
    mean[c] = mu;

    float var = (sq_sum * inv_M) - (mu * mu);
    var = std::max(var, 0.0f);

    float istd = 1.0f / std::sqrt(var + epsilon);
    inv_std[c] = istd;

    float unbiased_var = (M > 1) ? (var * M) / static_cast<float>(M - 1) : 0.0f;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;

    float g = affine ? gamma[c] : 1.0f;
    float b = beta[c];
    scale[c] = g * istd;
    bias_term[c] = b - (mu * scale[c]);
  }

  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float val = static_cast<float>(input[idx]) * scale[c] + bias_term[c];

      if (use_relu) {
        bool active = val > 0.0f;
        relu_mask[idx] = active;
        output[idx] = val * static_cast<float>(active);
      } else {
        output[idx] = static_cast<T>(val);
      }
    }
  });
}

template <typename T>
void run_backward_fused(const T *grad_output, const T *input, const float *mean,
                        const float *inv_std, const float *gamma, float *d_gamma, float *d_beta,
                        T *grad_input, const bool *relu_mask, size_t N, size_t C, size_t S,
                        bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);

  std::vector<float> partial_dy(N * C, 0.0f);
  std::vector<float> partial_dy_xn(N * C, 0.0f);

  parallel_for<size_t>(0, N, [&](size_t n) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C; ++c) {
        size_t idx = n * S * C + s * C + c;
        float dy = (!use_relu || relu_mask[idx]) ? static_cast<float>(grad_output[idx]) : 0.0f;
        float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];

        partial_dy[n * C + c] += dy;
        partial_dy_xn[n * C + c] += dy * x_hat;
      }
    }
  });

  std::vector<float> sum_dy(C, 0.0f);
  std::vector<float> sum_dy_xnorm(C, 0.0f);

  for (size_t c = 0; c < C; ++c) {
    float s_dy = 0.0f;
    float s_dy_xn = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      s_dy += partial_dy[n * C + c];
      s_dy_xn += partial_dy_xn[n * C + c];
    }

    sum_dy[c] = s_dy;
    sum_dy_xnorm[c] = s_dy_xn;

    if (affine) {
      d_gamma[c] += s_dy_xn;
      d_beta[c] += s_dy;
    }
  }

  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float g = gamma[c];
      float term1 = (g * inv_std[c]) * inv_M;

      float dy = (!use_relu || relu_mask[idx]) ? static_cast<float>(grad_output[idx]) : 0.0f;
      float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];

      float term2 = static_cast<float>(M) * dy - sum_dy[c] - x_hat * sum_dy_xnorm[c];
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

#define INSTANTIATE_BATCHNORM_NHWC(T)                                                           \
  template void compute_inference_output<T>(                                                    \
      const T *input, const float *running_mean, const float *running_var, const float *gamma,  \
      const float *beta, T *output, size_t N, size_t C, size_t S, float epsilon, bool affine);  \
                                                                                                \
  template void run_forward_fused<T>(                                                           \
      const T *input, float *mean, float *inv_std, float *running_mean, float *running_var,     \
      const float *gamma, const float *beta, T *output, bool *relu_mask, size_t N, size_t C,    \
      size_t S, float momentum, float epsilon, bool affine, bool use_relu);                     \
                                                                                                \
  template void run_backward_fused<T>(const T *grad_output, const T *input, const float *mean,  \
                                      const float *inv_std, const float *gamma, float *d_gamma, \
                                      float *d_beta, T *grad_input, const bool *relu_mask,      \
                                      size_t N, size_t C, size_t S, bool affine, bool use_relu);

INSTANTIATE_BATCHNORM_NHWC(fp16)
INSTANTIATE_BATCHNORM_NHWC(bf16)
INSTANTIATE_BATCHNORM_NHWC(float)
INSTANTIATE_BATCHNORM_NHWC(double)
#undef INSTANTIATE_BATCHNORM_NHWC

}  // namespace batchnorm_nhwc
}  // namespace cpu
}  // namespace tnn