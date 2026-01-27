#pragma once

#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>
namespace tnn {
namespace cuda {
namespace legacy_dense {

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_dense_forward_ex(const IO_T *input_data, const Param_T *weight_data, IO_T *output_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features, cudaStream_t stream);

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_weight_gradients_ex(const IO_T *input_data, const IO_T *gradient_data,
                                 Param_T *weight_grad_data, const size_t batch_size,
                                 const size_t input_features, const size_t output_features,
                                 cudaStream_t stream);

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_input_gradients_ex(const IO_T *gradient_data, const Param_T *weight_data,
                                IO_T *grad_input_data, const size_t batch_size,
                                const size_t input_features, const size_t output_features,
                                cudaStream_t stream);

template <typename IO_T, typename Param_T, typename Compute_T>
void compute_bias_gradients_ex(const IO_T *current_grad_data, Param_T *bias_gradient_data,
                               const size_t batch_size, const size_t output_features,
                               cudaStream_t stream);

template <typename IO_T, typename Param_T, typename Compute_T>
void add_bias_vector_ex(IO_T *output_data, const Param_T *bias_data, const size_t batch_size,
                        const size_t output_features, cudaStream_t stream);
} // namespace legacy_dense
} // namespace cuda
} // namespace tnn

#endif