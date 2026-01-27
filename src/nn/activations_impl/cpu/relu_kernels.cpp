#include "nn/activations_impl/cpu/relu_kernels.hpp"

#include "ops/cpu/kernels.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
template <typename T> void relu(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = input[i] > T(0) ? input[i] : T(0); });
}

template <typename T>
void relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size,
                       [&](size_t i) { grad_input[i] = input[i] > T(0) ? grad_output[i] : T(0); });
}

#define INSTANTIATE_RELU(T)                                                                        \
  template void relu<T>(const T *input, T *output, size_t size);                                   \
  template void relu_gradient<T>(const T *input, const T *grad_output, T *grad_input, size_t size);
INSTANTIATE_RELU(fp16)
INSTANTIATE_RELU(bf16)
INSTANTIATE_RELU(float)
INSTANTIATE_RELU(double)
#undef INSTANTIATE_RELUss

} // namespace cpu
} // namespace tnn