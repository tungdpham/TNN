#pragma once
#include <cstddef>

namespace tnn {
namespace cpu {
namespace class_token {
template <typename T>
void run_forward(const T *input, const T *token, T *output, size_t batch_size, size_t seq_len,
                 size_t embed_dim);

template <typename T>
void run_backward(const T *grad_output, T *grad_input, T *grad_token, size_t batch_size,
                  size_t seq_len, size_t embed_dim);
}  // namespace class_token
}  // namespace cpu
}  // namespace tnn
