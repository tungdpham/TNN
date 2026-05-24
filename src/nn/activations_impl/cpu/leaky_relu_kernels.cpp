#include "nn/activations_impl/cpu/leaky_relu_kernels.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
template <typename T>
void leaky_relu(const T *input, T *output, size_t size, T negative_slope) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = input[i] > T(0) ? input[i] : negative_slope * input[i];
  });
}

template <typename T>
void leaky_relu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size,
                         T negative_slope) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    grad_input[i] = input[i] > T(0) ? grad_output[i] : negative_slope * grad_output[i];
  });
}

#define INSTANTIATE(T)                                                                      \
  template void leaky_relu<T>(const T *input, T *output, size_t size, T negative_slope);    \
  template void leaky_relu_gradient<T>(const T *input, const T *grad_output, T *grad_input, \
                                       size_t size, T negative_slope);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace cpu
}  // namespace tnn
