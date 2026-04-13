#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace embedding {

template <typename T>
void run_forward(const T *input_data, const T *weight_data, T *output_data, size_t num_indices,
                 size_t vocab_size, size_t embed_dim, size_t padding_idx);

template <typename T>
void run_backward(const T *input_data, const T *gradient_data, T *weight_grad_data,
                  size_t num_indices, size_t vocab_size, size_t embed_dim, size_t padding_idx);

}  // namespace embedding
}  // namespace cpu
}  // namespace tnn
