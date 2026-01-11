#include "nn/layers_impl/cpu/class_token_ops.hpp"
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

  // grad_token usually accumulates across batch
  // But wait, if we are doing multi-threaded cpu, atomic?
  // Usually these kernels are single threaded per task or handle ranges.
  // The 'create_cpu_task' might invoke this in a thread.
  // However, `grad_token` is shared.
  // The previous implementation used `grad_token[token_offset] += ...` which implies accumulation.
  // If multiple threads run this for different N, it's a race condition unless we use atomics or
  // task handles sub-batches. Assuming standard single-threaded execution per layer for now or that
  // `create_cpu_task` handles it (it just runs the function). If `batch_size` > 1, we just
  // accumulate in a loop.

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

template void class_token_forward<float>(const float *input, const float *token, float *output,
                                         size_t batch_size, size_t seq_len, size_t embed_dim);
template void class_token_forward<double>(const double *input, const double *token, double *output,
                                          size_t batch_size, size_t seq_len, size_t embed_dim);

template void class_token_backward<float>(const float *grad_output, float *grad_input,
                                          float *grad_token, size_t batch_size, size_t seq_len,
                                          size_t embed_dim);
template void class_token_backward<double>(const double *grad_output, double *grad_input,
                                           double *grad_token, size_t batch_size, size_t seq_len,
                                           size_t embed_dim);

} // namespace cpu
} // namespace tnn
