#include "nn/activations_impl/cpu/sigmoid_kernels.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T> void sigmoid(const T *input, T *output, size_t size) {
  parallel_for<size_t>(
      0, size, [&](size_t i) { output[i] = T(1) / (T(1) + static_cast<T>(exp(-input[i]))); });
}

template <typename T>
void sigmoid_gradient(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T sigmoid_val = T(1) / (T(1) + static_cast<T>(exp(-input[i])));
    grad_input[i] = grad_output[i] * sigmoid_val * (T(1) - sigmoid_val);
  });
}

#define INSTANTIATE_SIGMOID(T)                                                                     \
  template void sigmoid<T>(const T *input, T *output, size_t size);                                \
  template void sigmoid_gradient<T>(const T *input, const T *grad_output, T *grad_input,           \
                                    size_t size);
INSTANTIATE_SIGMOID(fp16)
INSTANTIATE_SIGMOID(bf16)
INSTANTIATE_SIGMOID(float)
INSTANTIATE_SIGMOID(double)
#undef INSTANTIATE_SIGMOID

} // namespace cpu
} // namespace tnn
