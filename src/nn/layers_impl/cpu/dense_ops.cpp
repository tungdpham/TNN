/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/dense_ops.hpp"

#include "math/gemm.hpp"
#include "ops/ops.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {

template <typename T>
void compute_dense_forward(const T *input_data, const T *weight_data, T *output_data,
                           const size_t batch_size, const size_t input_features,
                           const size_t output_features) {
  cpu::gemm<T>(input_data, weight_data, output_data, batch_size, output_features, input_features,
               false, true);
}

template <typename T>
void compute_weight_gradients(const T *input_data, const T *gradient_data, T *weight_grad_data,
                              const size_t batch_size, const size_t input_features,
                              const size_t output_features) {
  cpu::gemm<T>(gradient_data, input_data, weight_grad_data, output_features, input_features,
               batch_size, true, false);
}

template <typename T>
void compute_input_gradients(const T *gradient_data, const T *weight_data, T *grad_input_data,
                             const size_t batch_size, const size_t input_features,
                             const size_t output_features) {
  std::fill_n(grad_input_data, batch_size * input_features, T(0));
  cpu::gemm<T>(gradient_data, weight_data, grad_input_data, batch_size, input_features,
               output_features, false, false);
}

template <typename T>
void compute_bias_gradients(const T *current_grad_data, T *bias_gradient_data,
                            const size_t batch_size, const size_t output_features) {
  parallel_for<size_t>(0, output_features, [&](size_t out_f) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      grad_sum += current_grad_data[n * output_features + out_f];
    }
    bias_gradient_data[out_f] += grad_sum;
  });
}

template <typename T>
void add_bias_vector(T *output_data, const T *bias_data, const size_t batch_size,
                     const size_t output_features) {
  parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] += bias_data[out_f];
  });
}

// Explicit template instantiations
template void compute_dense_forward<float>(const float *input_data, const float *weight_data,
                                           float *output_data, const size_t batch_size,
                                           const size_t input_features,
                                           const size_t output_features);
template void compute_dense_forward<double>(const double *input_data, const double *weight_data,
                                            double *output_data, const size_t batch_size,
                                            const size_t input_features,
                                            const size_t output_features);

template void compute_weight_gradients<float>(const float *input_data, const float *gradient_data,
                                              float *weight_grad_data, const size_t batch_size,
                                              const size_t input_features,
                                              const size_t output_features);
template void compute_weight_gradients<double>(const double *input_data,
                                               const double *gradient_data,
                                               double *weight_grad_data, const size_t batch_size,
                                               const size_t input_features,
                                               const size_t output_features);

template void compute_input_gradients<float>(const float *gradient_data, const float *weight_data,
                                             float *grad_input_data, const size_t batch_size,
                                             const size_t input_features,
                                             const size_t output_features);
template void compute_input_gradients<double>(const double *gradient_data,
                                              const double *weight_data, double *grad_input_data,
                                              const size_t batch_size, const size_t input_features,
                                              const size_t output_features);

template void compute_bias_gradients<float>(const float *current_grad_data,
                                            float *bias_gradient_data, const size_t batch_size,
                                            const size_t output_features);
template void compute_bias_gradients<double>(const double *current_grad_data,
                                             double *bias_gradient_data, const size_t batch_size,
                                             const size_t output_features);

template void add_bias_vector<float>(float *output_data, const float *bias_data,
                                     const size_t batch_size, const size_t output_features);
template void add_bias_vector<double>(double *output_data, const double *bias_data,
                                      const size_t batch_size, const size_t output_features);

// device_ptr implementations
template <typename T>
void compute_dense_forward(const device_ptr<T[]> &input_data, const device_ptr<T[]> &weight_data,
                           device_ptr<T[]> &output_data, const size_t batch_size,
                           const size_t input_features, const size_t output_features) {
  compute_dense_forward(input_data.get(), weight_data.get(), output_data.get(), batch_size,
                        input_features, output_features);
}

template <typename T>
void compute_weight_gradients(const device_ptr<T[]> &input_data,
                              const device_ptr<T[]> &gradient_data,
                              device_ptr<T[]> &weight_grad_data, const size_t batch_size,
                              const size_t input_features, const size_t output_features) {
  compute_weight_gradients(input_data.get(), gradient_data.get(), weight_grad_data.get(),
                           batch_size, input_features, output_features);
}

template <typename T>
void compute_input_gradients(const device_ptr<T[]> &gradient_data,
                             const device_ptr<T[]> &weight_data, device_ptr<T[]> &grad_input_data,
                             const size_t batch_size, const size_t input_features,
                             const size_t output_features) {
  compute_input_gradients(gradient_data.get(), weight_data.get(), grad_input_data.get(), batch_size,
                          input_features, output_features);
}

template <typename T>
void compute_bias_gradients(const device_ptr<T[]> &current_grad_data,
                            const device_ptr<T[]> &bias_gradient_data, const size_t batch_size,
                            const size_t output_features) {
  compute_bias_gradients(current_grad_data.get(), bias_gradient_data.get(), batch_size,
                         output_features);
}

template <typename T>
void add_bias_vector(device_ptr<T[]> &output_data, const device_ptr<T[]> &bias_data,
                     const size_t batch_size, const size_t output_features) {
  add_bias_vector(output_data.get(), bias_data.get(), batch_size, output_features);
}

// device_ptr explicit template instantiations
template void compute_dense_forward<float>(const device_ptr<float[]> &input_data,
                                           const device_ptr<float[]> &weight_data,
                                           device_ptr<float[]> &output_data,
                                           const size_t batch_size, const size_t input_features,
                                           const size_t output_features);

template void compute_weight_gradients<float>(const device_ptr<float[]> &input_data,
                                              const device_ptr<float[]> &gradient_data,
                                              device_ptr<float[]> &weight_grad_data,
                                              const size_t batch_size, const size_t input_features,
                                              const size_t output_features);

template void compute_input_gradients<float>(const device_ptr<float[]> &gradient_data,
                                             const device_ptr<float[]> &weight_data,
                                             device_ptr<float[]> &grad_input_data,
                                             const size_t batch_size, const size_t input_features,
                                             const size_t output_features);

template void compute_bias_gradients<float>(const device_ptr<float[]> &current_grad_data,
                                            const device_ptr<float[]> &bias_gradient_data,
                                            const size_t batch_size, const size_t output_features);

template void add_bias_vector<float>(device_ptr<float[]> &output_data,
                                     const device_ptr<float[]> &bias_data, const size_t batch_size,
                                     const size_t output_features);

} // namespace cpu
} // namespace tnn
