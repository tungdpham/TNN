#include "nn/activations_impl/cpu/relu_kernels.hpp"

#include "ops/cpu/kernels.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {
namespace cpu {
template <typename T> void relu(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = std::max(input[i], T(0)); });
}

template <typename T>
void relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size,
                       [&](size_t i) { grad_input[i] = input[i] > T(0) ? grad_output[i] : T(0); });
}

template void relu<float>(const float *input, float *output, size_t size);
template void relu<double>(const double *input, double *output, size_t size);

template void relu_gradient<float>(const float *input, const float *grad_output, float *grad_input,
                                   size_t size);
template void relu_gradient<double>(const double *input, const double *grad_output,
                                    double *grad_input, size_t size);

} // namespace cpu
} // namespace tnn