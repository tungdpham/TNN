/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/embedding_ops.hpp"
#include <cstring>

namespace tnn {
namespace cpu {
namespace embedding {

template <typename T>
void compute_embedding_forward(const T *input_data, const T *weight_data, T *output_data,
                               size_t num_indices, size_t vocab_size, size_t embed_dim,
                               size_t padding_idx) {
  for (size_t i = 0; i < num_indices; ++i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size) {
      idx = 0;
    }

    const T *w_row = weight_data + idx * embed_dim;
    T *out_row = output_data + i * embed_dim;
    std::memcpy(out_row, w_row, embed_dim * sizeof(T));
  }
}

template <typename T>
void compute_embedding_backward(const T *input_data, const T *gradient_data, T *weight_grad_data,
                                size_t num_indices, size_t vocab_size, size_t embed_dim,
                                size_t padding_idx) {
  for (size_t i = 0; i < num_indices; ++i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size)
      idx = 0;

    // Skip padding index updates if desired?
    if (padding_idx < vocab_size && idx == padding_idx)
      continue;

    const T *g_row = gradient_data + i * embed_dim;
    T *w_grad_row = weight_grad_data + idx * embed_dim;

    // CPU scatter add
    for (size_t j = 0; j < embed_dim; ++j) {
      w_grad_row[j] += g_row[j];
    }
  }
}

template void compute_embedding_forward<float>(const float *, const float *, float *, size_t,
                                               size_t, size_t, size_t);
template void compute_embedding_backward<float>(const float *, const float *, float *, size_t,
                                                size_t, size_t, size_t);

} // namespace embedding
} // namespace cpu
} // namespace tnn
