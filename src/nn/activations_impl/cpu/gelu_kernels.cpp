#include "nn/activations_impl/cpu/gelu_kernels.hpp"
#include <cmath>

namespace tnn {
namespace cpu {

template <typename T> void gelu(const T *input, T *output, size_t size) {
  constexpr T k0 = 0.7978845608028654; // sqrt(2/pi)
  constexpr T k1 = 0.044715;

  for (size_t i = 0; i < size; ++i) {
    T x = input[i];
    T x3 = x * x * x;
    T inner = k0 * (x + k1 * x3);
    output[i] = 0.5 * x * (1.0 + std::tanh(inner));
  }
}

template <typename T>
void gelu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  constexpr T k0 = 0.7978845608028654; // sqrt(2/pi)
  constexpr T k1 = 0.044715;

  for (size_t i = 0; i < size; ++i) {
    T x = input[i];
    T x3 = x * x * x;
    T inner = k0 * (x + k1 * x3);
    T tanh_inner = std::tanh(inner);

    T sechip = 1.0 / std::cosh(inner);
    T sechip2 = sechip * sechip;

    T d_inner_dx = k0 * (1.0 + 3.0 * k1 * x * x);

    // d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner_dx
    T grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sechip2 * d_inner_dx;

    grad_input[i] = grad_output[i] * grad;
  }
}

template void gelu<float>(const float *input, float *output, size_t size);
template void gelu<double>(const double *input, double *output, size_t size);
template void gelu_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                                   size_t size);
template void gelu_gradient<double>(const double *input, const double *grad_output,
                                    double *grad_input, size_t size);

} // namespace cpu
} // namespace tnn
