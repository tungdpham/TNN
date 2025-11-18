#include "nn/activations_impl/cpu/sigmoid_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T> void sigmoid(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = T(1) / (T(1) + std::exp(-input[i])); });
}

template <typename T> void sigmoid_gradient(const T *input, T *grad_output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T sigmoid_val = T(1) / (T(1) + std::exp(-input[i]));
    T local_grad = sigmoid_val * (T(1) - sigmoid_val);
    grad_output[i] *= local_grad;
  });
}

template void sigmoid<float>(const float *input, float *output, size_t size);
template void sigmoid<double>(const double *input, double *output, size_t size);

template void sigmoid_gradient<float>(const float *input, float *grad_output, size_t size);
template void sigmoid_gradient<double>(const double *input, double *grad_output, size_t size);

} // namespace cpu
} // namespace tnn
