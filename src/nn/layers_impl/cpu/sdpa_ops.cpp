/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/sdpa_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "type/type.hpp"

namespace synet {
namespace cpu {
namespace sdpa {
template <typename T>
void run_forward(const T *q, const T *k, const T *v, T *output,
                 typename TypeTraits<T>::ComputePrecision *scores, T *attn_weights,
                 size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                 float attn_scale, bool is_causal) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  // Shapes: Q, K, V: (batch, heads, seq_len, head_dim)
  size_t q_stride_b = num_heads * seq_len * head_dim;
  size_t q_stride_h = seq_len * head_dim;
  size_t q_stride_s = head_dim;

  // Compute attention scores: (B, H, S, S) = Q @ K^T * scale
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
          AccT score = AccT(0);
          for (size_t d = 0; d < head_dim; ++d) {
            size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
            size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
            score += static_cast<AccT>(q[q_idx]) * static_cast<AccT>(k[k_idx]);
          }
          score *= static_cast<AccT>(attn_scale);

          if (is_causal && j > i) {
            score = -std::numeric_limits<AccT>::infinity();
          }

          size_t score_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          scores[score_idx] = score;
        }
      }

      for (size_t i = 0; i < seq_len; ++i) {
        AccT max_score = -std::numeric_limits<AccT>::infinity();
        for (size_t j = 0; j < seq_len; ++j) {
          size_t score_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          if (std::isfinite(scores[score_idx])) {
            max_score = std::max(max_score, scores[score_idx]);
          }
        }

        AccT sum_exp = AccT(0);
        for (size_t j = 0; j < seq_len; ++j) {
          size_t score_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          AccT exp_val = std::isfinite(scores[score_idx])
                             ? static_cast<AccT>(std::exp(scores[score_idx] - max_score))
                             : AccT(0);
          size_t attn_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          attn_weights[attn_idx] = static_cast<T>(exp_val);
          sum_exp += exp_val;
        }

        AccT inv_sum = AccT(1) / (sum_exp + static_cast<AccT>(1e-9f));
        for (size_t j = 0; j < seq_len; ++j) {
          size_t attn_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          attn_weights[attn_idx] =
              static_cast<T>(static_cast<AccT>(attn_weights[attn_idx]) * inv_sum);
        }
      }
    }
  }

  // Compute output: O = Attention @ V
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t d = 0; d < head_dim; ++d) {
          AccT val = AccT(0);
          for (size_t j = 0; j < seq_len; ++j) {
            size_t attn_idx =
                b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
            size_t v_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
            val += static_cast<AccT>(attn_weights[attn_idx]) * static_cast<AccT>(v[v_idx]);
          }
          size_t o_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
          output[o_idx] = static_cast<T>(val);
        }
      }
    }
  }
}

template <typename T>
void run_backward(const T *q, const T *k, const T *v, const T *attn_weights, const T *grad_output,
                  typename TypeTraits<T>::ComputePrecision *grad_scores, T *grad_q, T *grad_k,
                  T *grad_v, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                  float attn_scale, bool is_causal) {
  using AccT = typename TypeTraits<T>::ComputePrecision;
  (void)is_causal;

  size_t q_stride_b = num_heads * seq_len * head_dim;
  size_t q_stride_h = seq_len * head_dim;
  size_t q_stride_s = head_dim;

  // grad_V = A^T @ grad_O
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t j = 0; j < seq_len; ++j) {
        for (size_t d = 0; d < head_dim; ++d) {
          AccT val = AccT(0);
          for (size_t i = 0; i < seq_len; ++i) {
            size_t attn_idx =
                b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
            size_t grad_o_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
            val += static_cast<AccT>(attn_weights[attn_idx]) *
                   static_cast<AccT>(grad_output[grad_o_idx]);
          }
          size_t v_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
          grad_v[v_idx] = static_cast<T>(val);
        }
      }
    }
  }

  // grad_scores <- grad_Attention = grad_O @ V^T
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
          AccT val = AccT(0);
          for (size_t d = 0; d < head_dim; ++d) {
            size_t grad_o_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
            size_t v_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
            val += static_cast<AccT>(grad_output[grad_o_idx]) * static_cast<AccT>(v[v_idx]);
          }
          size_t grad_scores_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          grad_scores[grad_scores_idx] = val;
        }
      }
    }
  }

  // Apply softmax Jacobian row-wise in-place to grad_scores.
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t i = 0; i < seq_len; ++i) {
        AccT sum_grad_scores_p = AccT(0);
        for (size_t j = 0; j < seq_len; ++j) {
          size_t attn_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          sum_grad_scores_p += static_cast<AccT>(attn_weights[attn_idx]) * grad_scores[attn_idx];
        }

        for (size_t j = 0; j < seq_len; ++j) {
          size_t attn_idx =
              b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
          grad_scores[attn_idx] = static_cast<AccT>(attn_weights[attn_idx]) *
                                  (grad_scores[attn_idx] - sum_grad_scores_p);
        }
      }
    }
  }

  // grad_Q = grad_scores @ K (scaled)
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t d = 0; d < head_dim; ++d) {
          AccT val = AccT(0);
          for (size_t j = 0; j < seq_len; ++j) {
            size_t grad_scores_idx =
                b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
            size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
            val += grad_scores[grad_scores_idx] * static_cast<AccT>(k[k_idx]);
          }
          size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
          grad_q[q_idx] = static_cast<T>(val * static_cast<AccT>(attn_scale));
        }
      }
    }
  }

  // grad_K = grad_scores^T @ Q (scaled)
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t j = 0; j < seq_len; ++j) {
        for (size_t d = 0; d < head_dim; ++d) {
          AccT val = AccT(0);
          for (size_t i = 0; i < seq_len; ++i) {
            size_t grad_scores_idx =
                b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) + i * seq_len + j;
            size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
            val += grad_scores[grad_scores_idx] * static_cast<AccT>(q[q_idx]);
          }
          size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
          grad_k[k_idx] = static_cast<T>(val * static_cast<AccT>(attn_scale));
        }
      }
    }
  }
}

#define INSTANTIATE(T)                                                                             \
  template void run_forward<T>(const T *, const T *, const T *, T *,                               \
                               typename TypeTraits<T>::ComputePrecision *, T *, size_t, size_t,    \
                               size_t, size_t, float, bool);                                       \
  template void run_backward<T>(const T *, const T *, const T *, const T *, const T *,             \
                                typename TypeTraits<T>::ComputePrecision *, T *, T *, T *, size_t, \
                                size_t, size_t, size_t, float, bool);
#include "macros/floating_type_instantiation.hpp"

#undef INSTANTIATE

}  // namespace sdpa
}  // namespace cpu
}  // namespace synet
