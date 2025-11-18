#include "nn/activations_impl/cpu/tanh_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T> void tanh(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = std::tanh(input[i]); });
}

template <typename T> void tanh_gradient(const T *input, T *grad_output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T tanh_val = std::tanh(input[i]);
    T local_grad = T(1) - tanh_val * tanh_val;
    grad_output[i] *= local_grad;
  });
}

template void tanh<float>(const float *input, float *output, size_t size);
template void tanh<double>(const double *input, double *output, size_t size);

template void tanh_gradient<float>(const float *input, float *grad_output, size_t size);
template void tanh_gradient<double>(const double *input, double *grad_output, size_t size);

} // namespace cpu
} // namespace tnn
