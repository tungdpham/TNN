/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace sdpa {
// Scaled Dot-Product Attention forward pass (CPU)
// Q, K, V: (batch, heads, seq_len, head_dim)
// Output: (batch, heads, seq_len, head_dim)
template <typename T>
void run_forward(const T *q, const T *k, const T *v, T *output, size_t batch_size, size_t num_heads,
                 size_t seq_len, size_t head_dim, float attn_scale, bool is_causal);

// Scaled Dot-Product Attention backward pass (CPU)
// Computes gradients w.r.t. Q, K, V
template <typename T>
void run_backward(const T *q, const T *k, const T *v, const T *output, const T *grad_output,
                  T *grad_q, T *grad_k, T *grad_v, size_t batch_size, size_t num_heads,
                  size_t seq_len, size_t head_dim, float attn_scale, bool is_causal);

}  // namespace sdpa
}  // namespace cpu
}  // namespace tnn
