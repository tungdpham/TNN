#pragma once

#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>
namespace tnn {
namespace cuda {
namespace dense {
template <typename T>
void compute_dense_forward(const T *input_data, const T *weight_data, T *output_data,
                           const size_t batch_size, const size_t input_features,
                           const size_t output_features, cudaStream_t stream);

template <typename T>
void compute_weight_gradients(const T *input_data, const T *gradient_data, T *weight_grad_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features, cudaStream_t stream);
template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *grad_input_data,
                             const size_t batch_size, const size_t input_features,
                             const size_t output_features, cudaStream_t stream);
template <typename T>
void compute_bias_gradients(const T *current_grad_data, const T *bias_gradient_data,
                            const size_t batch_size, const size_t output_features,
                            cudaStream_t stream);

template <typename T>
void add_bias_vector(T *output_data, const T *bias_data, const size_t batch_size,
                     const size_t output_features, cudaStream_t stream);
} // namespace dense
} // namespace cuda
} // namespace tnn

#endif