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

template <typename T> void elu_gradient(const T *input, T *grad_output, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T local_grad = input[i] > T(0) ? T(1) : alpha * std::exp(input[i]);
    grad_output[i] *= local_grad;
  });
}

template void elu<float>(const float *input, float *output, size_t size, float alpha);
template void elu<double>(const double *input, double *output, size_t size, double alpha);

template void elu_gradient<float>(const float *input, float *grad_output, size_t size, float alpha);
template void elu_gradient<double>(const double *input, double *grad_output, size_t size,
                                   double alpha);

} // namespace cpu
} // namespace tnn
