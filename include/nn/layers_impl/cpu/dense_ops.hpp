#pragma once

#include <cstddef>
namespace tnn {
namespace cpu {
namespace legacy_dense {

template <typename T>
void run_forward(const T *input_data, const T *weight_data, T *output_data, const size_t batch_size,
                 const size_t input_features, const size_t output_features);

template <typename T>
void run_wgrad(const T *input_data, const T *gradient_data, T *weight_grad_data,
               const size_t batch_size, const size_t input_features, const size_t output_features);

template <typename T>
void run_dgrad(const T *gradient_data, const T *weight_data, T *grad_input_data,
               const size_t batch_size, const size_t input_features, const size_t output_features);

template <typename T>
void run_bgrad(const T *current_grad_data, T *bias_gradient_data, const size_t batch_size,
               const size_t output_features);

template <typename T>
void add_bias(T *output_data, const T *bias_data, const size_t batch_size,
              const size_t output_features);
}  // namespace legacy_dense
}  // namespace cpu
}  // namespace tnn