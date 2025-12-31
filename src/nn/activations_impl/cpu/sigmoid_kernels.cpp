#include "nn/activations_impl/cpu/sigmoid_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T> void sigmoid(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = T(1) / (T(1) + std::exp(-input[i])); });
}

template <typename T>
void sigmoid_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T sigmoid_val = T(1) / (T(1) + std::exp(-input[i]));
    grad_input[i] = grad_output[i] * sigmoid_val * (T(1) - sigmoid_val);
  });
}

template void sigmoid<float>(const float *input, float *output, size_t size);
template void sigmoid<double>(const double *input, double *output, size_t size);

template void sigmoid_gradient<float>(const float *input, const float *grad_output,
                                      float *grad_input, size_t size);
template void sigmoid_gradient<double>(const double *input, const double *grad_output,
                                       double *grad_input, size_t size);

} // namespace cpu
} // namespace tnn
