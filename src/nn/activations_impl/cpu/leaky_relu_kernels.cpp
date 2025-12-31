#include "nn/activations_impl/cpu/leaky_relu_kernels.hpp"

#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {
template <typename T> void leaky_relu(const T *input, T *output, size_t size, T negative_slope) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = input[i] > T(0) ? input[i] : negative_slope * input[i];
  });
}

template <typename T>
void leaky_relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size,
                         T negative_slope) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    grad_input[i] = input[i] > T(0) ? grad_output[i] : negative_slope * grad_output[i];
  });
}

template void leaky_relu<float>(const float *input, float *output, size_t size,
                                float negative_slope);
template void leaky_relu<double>(const double *input, double *output, size_t size,
                                 double negative_slope);

template void leaky_relu_gradient<float>(const float *input, const float *grad_output,
                                         float *grad_input, size_t size, float negative_slope);
template void leaky_relu_gradient<double>(const double *input, const double *grad_output,
                                          double *grad_input, size_t size, double negative_slope);

} // namespace cpu
} // namespace tnn
