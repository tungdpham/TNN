/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/batchnorm_nhwc_ops.hpp"

#include <cmath>
#include <vector>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace batchnorm_nhwc {

// Inference forward: normalize each element using running statistics.
// NHWC layout: element [n, s, c] lives at index n*S*C + s*C + c.
template <typename T>
void compute_inference_output(const T *input, const float *running_mean, const float *running_var,
                              const float *gamma, const float *beta, T *output, size_t N, size_t C,
                              size_t S, float epsilon, bool affine) {
  parallel_for_2d<size_t>(N, C, [&](size_t n, size_t c) {
    const float inv_std = 1.0f / std::sqrt(running_var[c] + epsilon);
    const float mean_val = running_mean[c];
    const float g = affine ? gamma[c] : 1.0f;
    const float b = affine ? beta[c] : 0.0f;

    for (size_t s = 0; s < S; ++s) {
      const size_t idx = n * S * C + s * C + c;
      const float x_hat = (static_cast<float>(input[idx]) - mean_val) * inv_std;
      output[idx] = static_cast<T>(g * x_hat + b);
    }
  });
}

// Training forward: compute per-channel statistics, update running stats, normalize.
template <typename T>
void run_forward_fused(const T *input, float *mean, float *inv_std, float *running_mean,
                       float *running_var, const float *gamma, const float *beta, T *output,
                       bool *relu_mask, size_t N, size_t C, size_t S, float momentum,
                       float epsilon, bool affine, bool use_relu) {
  const size_t M = N * S;
  const float inv_M = 1.0f / static_cast<float>(M);

  // Pass 1 (per channel): compute mean, variance, inv_std, and update running statistics.
  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      for (size_t s = 0; s < S; ++s) {
        sum += static_cast<float>(input[n * S * C + s * C + c]);
      }
    }
    const float mu = sum * inv_M;
    mean[c] = mu;

    float var_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      for (size_t s = 0; s < S; ++s) {
        const float diff = static_cast<float>(input[n * S * C + s * C + c]) - mu;
        var_sum += diff * diff;
      }
    }
    inv_std[c] = 1.0f / std::sqrt(var_sum * inv_M + epsilon);

    // Unbiased variance for running stats (Bessel's correction).
    const float unbiased_var = (M > 1) ? var_sum / static_cast<float>(M - 1) : 0.0f;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  });

  // Pass 2 (per sample × channel): normalize, apply affine transform, and optionally ReLU.
  parallel_for_2d<size_t>(N, C, [&](size_t n, size_t c) {
    const float mu = mean[c];
    const float istd = inv_std[c];
    const float g = affine ? gamma[c] : 1.0f;
    const float b = affine ? beta[c] : 0.0f;

    for (size_t s = 0; s < S; ++s) {
      const size_t idx = n * S * C + s * C + c;
      const float val = g * ((static_cast<float>(input[idx]) - mu) * istd) + b;
      if (use_relu) {
        const bool active = val > 0.0f;
        relu_mask[idx] = active;
        output[idx] = static_cast<T>(active ? val : 0.0f);
      } else {
        output[idx] = static_cast<T>(val);
      }
    }
  });
}

// Backward: accumulate d_gamma/d_beta and compute grad_input.
// Recomputes x_hat on the fly from input, mean, and inv_std.
// When use_relu is true the incoming gradient is gated by relu_mask before the BN backward.
template <typename T>
void run_backward_fused(const T *grad_output, const T *input, const float *mean,
                        const float *inv_std, const float *gamma, float *d_gamma, float *d_beta,
                        T *grad_input, const bool *relu_mask, size_t N, size_t C, size_t S,
                        bool affine, bool use_relu) {
  const size_t M = N * S;
  const float inv_M = 1.0f / static_cast<float>(M);

  // Temporary per-channel sums needed by the batch-norm backward formula, kept separate from
  // the gradient accumulators so that gradient accumulation across micro-batches stays correct.
  std::vector<float> sum_dy(C, 0.0f);
  std::vector<float> sum_dy_xnorm(C, 0.0f);

  // Pass 1 (per channel): gate grad through ReLU mask, accumulate sum_dy and sum_dy*x_hat.
  parallel_for<size_t>(0, C, [&](size_t c) {
    const float mu = mean[c];
    const float istd = inv_std[c];
    float s_dy = 0.0f;
    float s_dy_xn = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      for (size_t s = 0; s < S; ++s) {
        const size_t idx = n * S * C + s * C + c;
        const float dy = (use_relu && !relu_mask[idx])
                             ? 0.0f
                             : static_cast<float>(grad_output[idx]);
        const float x_hat = (static_cast<float>(input[idx]) - mu) * istd;
        s_dy += dy;
        s_dy_xn += dy * x_hat;
      }
    }
    sum_dy[c] = s_dy;
    sum_dy_xnorm[c] = s_dy_xn;
    if (affine) {
      d_gamma[c] += s_dy_xn;
      d_beta[c] += s_dy;
    }
  });

  // Pass 2 (per sample × channel): compute grad_input using the BN backward formula:
  //   grad_x = (g * inv_std / M) * (M*dy - sum_dy - x_hat * sum_dy_x_hat)
  parallel_for_2d<size_t>(N, C, [&](size_t n, size_t c) {
    const float mu = mean[c];
    const float g = (affine && gamma) ? gamma[c] : 1.0f;
    const float istd = inv_std[c];
    const float s_dy = sum_dy[c];
    const float s_dy_xn = sum_dy_xnorm[c];
    const float term1 = (g * istd) * inv_M;

    for (size_t s = 0; s < S; ++s) {
      const size_t idx = n * S * C + s * C + c;
      const float dy = (use_relu && !relu_mask[idx])
                           ? 0.0f
                           : static_cast<float>(grad_output[idx]);
      const float x_hat = (static_cast<float>(input[idx]) - mu) * istd;
      const float term2 = static_cast<float>(M) * dy - s_dy - x_hat * s_dy_xn;
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

#define INSTANTIATE_BATCHNORM_NHWC(T)                                                          \
  template void compute_inference_output<T>(                                                   \
      const T *input, const float *running_mean, const float *running_var, const float *gamma, \
      const float *beta, T *output, size_t N, size_t C, size_t S, float epsilon, bool affine); \
                                                                                               \
  template void run_forward_fused<T>(                                                          \
      const T *input, float *mean, float *inv_std, float *running_mean, float *running_var,    \
      const float *gamma, const float *beta, T *output, bool *relu_mask, size_t N, size_t C,  \
      size_t S, float momentum, float epsilon, bool affine, bool use_relu);                    \
                                                                                               \
  template void run_backward_fused<T>(                                                         \
      const T *grad_output, const T *input, const float *mean, const float *inv_std,           \
      const float *gamma, float *d_gamma, float *d_beta, T *grad_input,                       \
      const bool *relu_mask, size_t N, size_t C, size_t S, bool affine, bool use_relu);

INSTANTIATE_BATCHNORM_NHWC(fp16)
INSTANTIATE_BATCHNORM_NHWC(bf16)
INSTANTIATE_BATCHNORM_NHWC(float)
INSTANTIATE_BATCHNORM_NHWC(double)
#undef INSTANTIATE_BATCHNORM_NHWC

}  // namespace batchnorm_nhwc
}  // namespace cpu
}  // namespace tnn
