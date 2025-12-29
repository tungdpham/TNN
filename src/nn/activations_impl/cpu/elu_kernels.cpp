#include "nn/activations_impl/cpu/elu_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T> void elu(const T *input, T *output, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = input[i] > T(0) ? input[i] : alpha * (std::exp(input[i]) - T(1));
  });
}

template <typename T>
void elu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    grad_input[i] = input[i] > T(0) ? grad_output[i] : grad_output[i] * alpha * std::exp(input[i]);
  });
}

template void elu<float>(const float *input, float *output, size_t size, float alpha);
template void elu<double>(const double *input, double *output, size_t size, double alpha);

template void elu_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                                  size_t size, float alpha);
template void elu_gradient<double>(const double *input, const double *grad_output,
                                   double *grad_input, size_t size, double alpha);

} // namespace cpu
} // namespace tnn
