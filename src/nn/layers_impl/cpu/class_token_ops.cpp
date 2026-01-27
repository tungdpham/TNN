#include "nn/layers_impl/cpu/class_token_ops.hpp"

#include "type/type.hpp"
#include <cstring>

namespace tnn {
namespace cpu {

template <typename T>
void class_token_forward(const T *input, const T *token, T *output, size_t batch_size,
                         size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;

  for (size_t n = 0; n < batch_size; ++n) {
    // Current pointer in output for this sequence
    T *out_seq = output + n * output_S * E;
    // Current pointer in input for this sequence
    const T *in_seq = input + n * S * E;

    // 1. Copy class token to the first position (s=0)
    std::memcpy(out_seq, token, E * sizeof(T));

    // 2. Copy input to the rest (s=1..S)
    std::memcpy(out_seq + E, in_seq, S * E * sizeof(T));
  }
}

template <typename T>
void class_token_backward(const T *grad_output, T *grad_input, T *grad_token, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;

  for (size_t n = 0; n < batch_size; ++n) {
    const T *grad_out_seq = grad_output + n * output_S * E;
    T *grad_in_seq = grad_input + n * S * E;

    // 1. Accumulate gradients for class token from first position
    const T *grad_token_part = grad_out_seq;
    for (size_t e = 0; e < E; ++e) {
      grad_token[e] += grad_token_part[e];
    }

    // 2. Copy gradients to input from the rest
    std::memcpy(grad_in_seq, grad_out_seq + E, S * E * sizeof(T));
  }
}

#define INSTANTIATE_CLASS_TOKEN(T)                                                                 \
  template void class_token_forward<T>(const T *input, const T *token, T *output,                  \
                                       size_t batch_size, size_t seq_len, size_t embed_dim);       \
                                                                                                   \
  template void class_token_backward<T>(const T *grad_output, T *grad_input, T *grad_token,        \
                                        size_t batch_size, size_t seq_len, size_t embed_dim);
INSTANTIATE_CLASS_TOKEN(fp16)
INSTANTIATE_CLASS_TOKEN(bf16)
INSTANTIATE_CLASS_TOKEN(float)
INSTANTIATE_CLASS_TOKEN(double)
#undef INSTANTIATE_CLASS_TOKEN

} // namespace cpu
} // namespace tnn
