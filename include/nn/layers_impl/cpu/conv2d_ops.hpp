#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
template <typename T>
void compute_conv_forward(const T *col_data, const T *weight_data, T *output_data,
                          const size_t output_size, const size_t kernel_size,
                          const size_t out_channels);
template <typename T>
void compute_weight_gradients(const T *col_data, const T *gradient_data, T *weight_grad_data,
                              const size_t output_size, const size_t kernel_size,
                              const size_t out_channels);

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *col_grad_data,
                             const size_t output_size, const size_t kernel_size,
                             const size_t out_channels);

template <typename T>
void compute_bias_gradients(const T *gradient_data, T *bias_grad_data, const size_t batch_size,
                            const size_t output_h, const size_t output_w,
                            const size_t out_channels);

template <typename T>
void add_bias_to_output(T *output_data, const T *bias_data, const size_t batch_size,
                        const size_t output_h, const size_t output_w, const size_t out_channels);
} // namespace cpu
} // namespace tnn