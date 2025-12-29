#include "nn/activations_impl/cpu/tanh_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T> void tanh(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = std::tanh(input[i]); });
}

template <typename T>
void tanh_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T tanh_val = std::tanh(input[i]);
    grad_input[i] = grad_output[i] * (T(1) - tanh_val * tanh_val);
  });
}

template void tanh<float>(const float *input, float *output, size_t size);
template void tanh<double>(const double *input, double *output, size_t size);

template void tanh_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                                   size_t size);
template void tanh_gradient<double>(const double *input, const double *grad_output,
                                    double *grad_input, size_t size);

} // namespace cpu
} // namespace tnn
