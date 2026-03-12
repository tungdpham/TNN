/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/embedding_ops.hpp"

#include <algorithm>
#include <cstring>

#include "type/type.hpp"

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

    T *out_row = output_data + i * embed_dim;

    if (padding_idx < vocab_size && idx == padding_idx) {
      std::fill(out_row, out_row + embed_dim, T(0));
      continue;
    }

    const T *w_row = weight_data + idx * embed_dim;
    std::memcpy(out_row, w_row, embed_dim * sizeof(T));
  }
}

template <typename T>
void compute_embedding_backward(const T *input_data, const T *gradient_data, T *weight_grad_data,
                                size_t num_indices, size_t vocab_size, size_t embed_dim,
                                size_t padding_idx) {
  for (size_t i = 0; i < num_indices; ++i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size) idx = 0;

    // Skip padding index updates if desired?
    if (padding_idx < vocab_size && idx == padding_idx) continue;

    const T *g_row = gradient_data + i * embed_dim;
    T *w_grad_row = weight_grad_data + idx * embed_dim;

    // CPU scatter add
    for (size_t j = 0; j < embed_dim; ++j) {
      w_grad_row[j] += g_row[j];
    }
  }
}

#define INSTANTIATE_EMBEDDING(T)                                                                 \
  template void compute_embedding_forward<T>(const T *, const T *, T *, size_t, size_t, size_t,  \
                                             size_t);                                            \
                                                                                                 \
  template void compute_embedding_backward<T>(const T *, const T *, T *, size_t, size_t, size_t, \
                                              size_t);
INSTANTIATE_EMBEDDING(fp16)
INSTANTIATE_EMBEDDING(bf16)
INSTANTIATE_EMBEDDING(float)
INSTANTIATE_EMBEDDING(double)
#undef INSTANTIATE_EMBEDDING

}  // namespace embedding
}  // namespace cpu
}  // namespace tnn
