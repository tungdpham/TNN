#include "nn/activations_impl/cpu/elu_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
template <typename T>
void elu(const T *input, T *output, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = input[i] > T(0) ? input[i] : alpha * (static_cast<T>(exp(input[i])) - T(1));
  });
}

template <typename T>
void elu_gradient(const T *input, const T *grad_output, T *grad_input, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    grad_input[i] =
        input[i] > T(0) ? grad_output[i] : grad_output[i] * alpha * static_cast<T>(exp(input[i]));
  });
}

#define INSTANTIATE_ELU(T)                                                                        \
  template void elu<T>(const T *input, T *output, size_t size, T alpha);                          \
  template void elu_gradient<T>(const T *input, const T *grad_output, T *grad_input, size_t size, \
                                T alpha);
INSTANTIATE_ELU(fp16)
INSTANTIATE_ELU(bf16)
INSTANTIATE_ELU(float)
INSTANTIATE_ELU(double)
#undef INSTANTIATE_ELU

}  // namespace cpu
}  // namespace tnn
