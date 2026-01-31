#include "nn/activations_impl/cpu/tanh_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
template <typename T>
void tanh(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = static_cast<T>(std::tanh(static_cast<double>(input[i])));
  });
}

template <typename T>
void tanh_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    double tanh_val = std::tanh(static_cast<double>(input[i]));
    grad_input[i] =
        static_cast<T>(static_cast<double>(grad_output[i]) * (1.0 - tanh_val * tanh_val));
  });
}

#define INSTANTIATE_TANH(T)                                      \
  template void tanh<T>(const T *input, T *output, size_t size); \
  template void tanh_gradient<T>(const T *input, const T *grad_output, T *grad_input, size_t size);

INSTANTIATE_TANH(fp16)
INSTANTIATE_TANH(bf16)
INSTANTIATE_TANH(float)
INSTANTIATE_TANH(double)
#undef INSTANTIATE_TANH

}  // namespace cpu
}  // namespace tnn
