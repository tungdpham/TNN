#include "nn/activations_impl/cpu/gelu_kernels.hpp"
#include "type/type.hpp"
#include <cmath>

namespace tnn {
namespace cpu {

template <typename T> void gelu(const T *input, T *output, size_t size) {
  const double k0 = 0.7978845608028654; // sqrt(2/pi)
  const double k1 = 0.044715;

  for (size_t i = 0; i < size; ++i) {
    double x = static_cast<double>(input[i]);
    double x3 = x * x * x;
    double inner = k0 * (x + k1 * x3);
    output[i] = static_cast<T>(0.5 * x * (1.0 + std::tanh(inner)));
  }
}

template <typename T>
void gelu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  const double k0 = 0.7978845608028654; // sqrt(2/pi)
  const double k1 = 0.044715;

  for (size_t i = 0; i < size; ++i) {
    double x = static_cast<double>(input[i]);
    double x3 = x * x * x;
    double inner = k0 * (x + k1 * x3);
    double tanh_inner = std::tanh(inner);

    double sechip = 1.0 / std::cosh(inner);
    double sechip2 = sechip * sechip;

    double d_inner_dx = k0 * (1.0 + 3.0 * k1 * x * x);

    // d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner_dx
    double grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sechip2 * d_inner_dx;

    grad_input[i] = static_cast<T>(static_cast<double>(grad_output[i]) * grad);
  }
}

#define INSTANTIATE_GELU(T)                                                                        \
  template void gelu<T>(const T *input, T *output, size_t size);                                   \
  template void gelu_gradient<T>(const T *input, const T *grad_output, T *grad_input, size_t size);
INSTANTIATE_GELU(fp16)
INSTANTIATE_GELU(bf16)
INSTANTIATE_GELU(float)
INSTANTIATE_GELU(double)
#undef INSTANTIATE_GELU

} // namespace cpu
} // namespace tnn
