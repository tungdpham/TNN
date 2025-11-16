#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
template <typename T>
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size);

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size);

template <typename T>
void normalize_and_scale_optimized(const T *input_data, const T *mean_data, const T *std_data,
                                   const T *gamma_data, const T *beta_data, T *output_data,
                                   T *normalized_data, size_t batch_size, size_t channels,
                                   size_t spatial_size, bool affine);

template <typename T>
void compute_affine_gradients_optimized(const T *gradient_data, const T *normalized_data,
                                        T *gamma_grad, T *beta_grad, size_t batch_size,
                                        size_t channels, size_t spatial_size);

} // namespace cpu
} // namespace tnn