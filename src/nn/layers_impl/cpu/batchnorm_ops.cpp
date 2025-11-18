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
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size) {
  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);

  parallel_for<size_t>(0, channels, [&](size_t c) {
    T sum = T(0);
    const size_t channel_stride = channels * spatial_size;
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;
      sum += ops::cpu::sum(batch_channel_ptr, spatial_size);
    }

    mean_data[c] = sum * inv_total;
  });
}

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size) {
  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);

  parallel_for<size_t>(0, channels, [&](size_t c) {
    T sum_sq = T(0);
    const T mean_val = mean_data[c];
    const size_t channel_stride = channels * spatial_size;
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;

      // Use AVX2-optimized sum of squared differences
      sum_sq += ops::cpu::sum_squared_diff(batch_channel_ptr, mean_val, spatial_size);
    }

    var_data[c] = sum_sq * inv_total;
  });
}

template <typename T>
void normalize_and_scale_optimized(const T *input_data, const T *mean_data, const T *std_data,
                                   const T *gamma_data, const T *beta_data, T *output_data,
                                   T *normalized_data, size_t batch_size, size_t channels,
                                   size_t spatial_size, bool affine) {
  const size_t channel_stride = channels * spatial_size;

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const T mean_val = mean_data[c];
    const T std_val = std_data[c];
    const T inv_std = T(1) / std_val;

    const size_t n_offset = n * channel_stride;
    const size_t c_offset = c * spatial_size;
    const size_t base_idx = n_offset + c_offset;

    const T *input_ptr = input_data + base_idx;
    T *normalized_ptr = normalized_data + base_idx;
    T *output_ptr = output_data + base_idx;

    // Normalize: (x - mean) / std - vectorized with AVX2
    ops::cpu::sub_mul_scalar(input_ptr, mean_val, inv_std, normalized_ptr, spatial_size);

    if (affine) {
      const T gamma_val = gamma_data[c];
      const T beta_val = beta_data[c];

      // Scale and shift: gamma * normalized + beta - vectorized FMA with AVX2
      ops::cpu::mul_add_scalar(normalized_ptr, gamma_val, beta_val, output_ptr, spatial_size);
    } else {
      ops::cpu::copy(normalized_ptr, output_ptr, spatial_size);
    }
  });
}

template <typename T>
void compute_affine_gradients_optimized(const T *gradient_data, const T *normalized_data,
                                        T *gamma_grad, T *beta_grad, size_t batch_size,
                                        size_t channels, size_t spatial_size) {
  const size_t channel_stride = channels * spatial_size;

  parallel_for<size_t>(0, channels, [&](size_t c) {
    T gamma_sum = T(0);
    T beta_sum = T(0);
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const size_t base_idx = n * channel_stride + c_offset;
      const T *grad_ptr = gradient_data + base_idx;
      const T *norm_ptr = normalized_data + base_idx;

      for (size_t i = 0; i < spatial_size; ++i) {
        gamma_sum += grad_ptr[i] * norm_ptr[i];
        beta_sum += grad_ptr[i];
      }
    }

    gamma_grad[c] += gamma_sum;
    beta_grad[c] += beta_sum;
  });
}

// Explicit template instantiations
template void compute_channel_mean<float>(const float *input_data, float *mean_data,
                                          size_t batch_size, size_t channels, size_t spatial_size);
template void compute_channel_mean<double>(const double *input_data, double *mean_data,
                                           size_t batch_size, size_t channels, size_t spatial_size);

template void compute_channel_variance<float>(const float *input_data, const float *mean_data,
                                              float *var_data, size_t batch_size, size_t channels,
                                              size_t spatial_size);
template void compute_channel_variance<double>(const double *input_data, const double *mean_data,
                                               double *var_data, size_t batch_size, size_t channels,
                                               size_t spatial_size);

template void normalize_and_scale_optimized<float>(const float *input_data, const float *mean_data,
                                                   const float *std_data, const float *gamma_data,
                                                   const float *beta_data, float *output_data,
                                                   float *normalized_data, size_t batch_size,
                                                   size_t channels, size_t spatial_size,
                                                   bool affine);
template void normalize_and_scale_optimized<double>(
    const double *input_data, const double *mean_data, const double *std_data,
    const double *gamma_data, const double *beta_data, double *output_data, double *normalized_data,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine);

template void compute_affine_gradients_optimized<float>(const float *gradient_data,
                                                        const float *normalized_data,
                                                        float *gamma_grad, float *beta_grad,
                                                        size_t batch_size, size_t channels,
                                                        size_t spatial_size);
template void compute_affine_gradients_optimized<double>(const double *gradient_data,
                                                         const double *normalized_data,
                                                         double *gamma_grad, double *beta_grad,
                                                         size_t batch_size, size_t channels,
                                                         size_t spatial_size);

template <typename T>
void compute_batch_std(const T *batch_var_data, T *batch_std_data, size_t channels, T epsilon) {
  for (size_t c = 0; c < channels; ++c) {
    batch_std_data[c] = std::sqrt(batch_var_data[c] + epsilon);
  }
}

template <typename T>
void update_running_stats(T *running_mean_data, T *running_var_data, const T *batch_mean_data,
                          const T *batch_var_data, size_t channels, T momentum) {
  parallel_for<size_t>(0, channels, [&](size_t c) {
    running_mean_data[c] = (T(1) - momentum) * running_mean_data[c] + momentum * batch_mean_data[c];
    running_var_data[c] = (T(1) - momentum) * running_var_data[c] + momentum * batch_var_data[c];
  });
}

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

template <typename T>
void compute_grad_normalized(const T *gradient_data, const T *gamma_data, T *grad_normalized_data,
                             size_t batch_size, size_t channels, size_t spatial_size, bool affine) {
  const size_t channel_stride = channels * spatial_size;

  if (affine) {
    parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
      const T gamma_val = gamma_data[c];
      const size_t base_idx = n * channel_stride + c * spatial_size;

      const T *grad_ptr = gradient_data + base_idx;
      T *grad_norm_ptr = grad_normalized_data + base_idx;

      ops::cpu::mul_scalar(grad_ptr, gamma_val, grad_norm_ptr, spatial_size);
    });
  } else {
    ops::cpu::copy(gradient_data, grad_normalized_data, batch_size * channels * spatial_size);
  }
}

template <typename T>
void compute_backward_sums(const T *grad_normalized_data, const T *normalized_data,
                           T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data,
                           size_t batch_size, size_t channels, size_t spatial_size) {
  const size_t channel_stride = channels * spatial_size;

  parallel_for<size_t>(0, channels, [&](size_t c) {
    T sum_grad_norm = T(0);
    T sum_grad_norm_x_norm = T(0);
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const size_t base_idx = n * channel_stride + c_offset;
      const T *grad_norm_ptr = grad_normalized_data + base_idx;
      const T *norm_ptr = normalized_data + base_idx;

      for (size_t i = 0; i < spatial_size; ++i) {
        sum_grad_norm += grad_norm_ptr[i];
        sum_grad_norm_x_norm += grad_norm_ptr[i] * norm_ptr[i];
      }
    }

    sum_grad_normalized_data[c] = sum_grad_norm;
    sum_grad_norm_times_norm_data[c] = sum_grad_norm_x_norm;
  });
}

template <typename T>
void compute_input_gradients_batchnorm(const T *grad_normalized_data, const T *normalized_data,
                                       const T *std_data, const T *sum_grad_normalized_data,
                                       const T *sum_grad_norm_times_norm_data, T *grad_input_data,
                                       size_t batch_size, size_t channels, size_t spatial_size,
                                       size_t total_elements) {
  const size_t channel_stride = channels * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);

  parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    const T std_val_c = std_data[c];
    const T inv_std = T(1) / std_val_c;
    const T sum_grad_norm = sum_grad_normalized_data[c];
    const T sum_grad_norm_x_norm = sum_grad_norm_times_norm_data[c];

    const size_t base_idx = n * channel_stride + c * spatial_size;
    const T *grad_norm_ptr = grad_normalized_data + base_idx;
    const T *norm_ptr = normalized_data + base_idx;
    T *grad_input_ptr = grad_input_data + base_idx;

    for (size_t i = 0; i < spatial_size; ++i) {
      // ∂L/∂x = (1/N) * (1/σ) * [N * ∂L/∂x̂ - Σ(∂L/∂x̂) - x̂ * Σ(∂L/∂x̂ * x̂)]
      grad_input_ptr[i] = inv_std * inv_total *
                          (static_cast<T>(total_elements) * grad_norm_ptr[i] - sum_grad_norm -
                           norm_ptr[i] * sum_grad_norm_x_norm);
    }
  });
}

template void compute_batch_std<float>(const float *batch_var_data, float *batch_std_data,
                                       size_t channels, float epsilon);
template void compute_batch_std<double>(const double *batch_var_data, double *batch_std_data,
                                        size_t channels, double epsilon);

template void update_running_stats<float>(float *running_mean_data, float *running_var_data,
                                          const float *batch_mean_data, const float *batch_var_data,
                                          size_t channels, float momentum);
template void update_running_stats<double>(double *running_mean_data, double *running_var_data,
                                           const double *batch_mean_data,
                                           const double *batch_var_data, size_t channels,
                                           double momentum);

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

template void compute_grad_normalized<float>(const float *gradient_data, const float *gamma_data,
                                             float *grad_normalized_data, size_t batch_size,
                                             size_t channels, size_t spatial_size, bool affine);
template void compute_grad_normalized<double>(const double *gradient_data, const double *gamma_data,
                                              double *grad_normalized_data, size_t batch_size,
                                              size_t channels, size_t spatial_size, bool affine);

template void compute_backward_sums<float>(const float *grad_normalized_data,
                                           const float *normalized_data,
                                           float *sum_grad_normalized_data,
                                           float *sum_grad_norm_times_norm_data, size_t batch_size,
                                           size_t channels, size_t spatial_size);
template void compute_backward_sums<double>(const double *grad_normalized_data,
                                            const double *normalized_data,
                                            double *sum_grad_normalized_data,
                                            double *sum_grad_norm_times_norm_data,
                                            size_t batch_size, size_t channels,
                                            size_t spatial_size);

template void compute_input_gradients_batchnorm<float>(
    const float *grad_normalized_data, const float *normalized_data, const float *std_data,
    const float *sum_grad_normalized_data, const float *sum_grad_norm_times_norm_data,
    float *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements);
template void compute_input_gradients_batchnorm<double>(
    const double *grad_normalized_data, const double *normalized_data, const double *std_data,
    const double *sum_grad_normalized_data, const double *sum_grad_norm_times_norm_data,
    double *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements);

} // namespace batchnorm
} // namespace cpu
} // namespace tnn
