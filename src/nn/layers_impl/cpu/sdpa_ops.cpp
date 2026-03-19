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

namespace tnn {
namespace cpu {

template <typename T>
void sdpa_forward(const T *q, const T *k, const T *v, T *output, size_t batch_size,
                  size_t num_heads, size_t seq_len, size_t head_dim, float attn_scale,
                  bool is_causal) {
  // Shapes: Q, K, V: (batch, heads, seq_len, head_dim)
  const size_t q_stride_b = num_heads * seq_len * head_dim;
  const size_t q_stride_h = seq_len * head_dim;
  const size_t q_stride_s = head_dim;

  // Allocate temporary attention scores and softmax
  T *scores = new T[batch_size * num_heads * seq_len * seq_len];
  T *attn_weights = new T[batch_size * num_heads * seq_len * seq_len];

  try {
    // Compute attention scores: (B, H, S, S) = Q @ K^T / sqrt(d)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
          for (size_t j = 0; j < seq_len; ++j) {
            // Compute dot product: q[b,h,i,:] · k[b,h,j,:]
            T score = 0;
            for (size_t d = 0; d < head_dim; ++d) {
              size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
              size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
              score += q[q_idx] * k[k_idx];
            }
            score *= static_cast<T>(attn_scale);

            // Apply causal mask if needed
            if (is_causal && j > i) {
              score = -std::numeric_limits<T>::infinity();
            }

            size_t score_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                               i * seq_len + j;
            scores[score_idx] = score;
          }
        }

        // Softmax over seq_len dimension
        for (size_t i = 0; i < seq_len; ++i) {
          // Find max for numerical stability
          T max_score = -std::numeric_limits<T>::infinity();
          for (size_t j = 0; j < seq_len; ++j) {
            size_t score_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                               i * seq_len + j;
            if (std::isfinite(scores[score_idx])) {
              max_score = std::max(max_score, scores[score_idx]);
            }
          }

          // Compute exp and sum
          T sum_exp = 0;
          for (size_t j = 0; j < seq_len; ++j) {
            size_t score_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                               i * seq_len + j;
            T exp_val = std::isfinite(scores[score_idx]) ? std::exp(scores[score_idx] - max_score)
                                                         : 0;
            size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                              i * seq_len + j;
            attn_weights[attn_idx] = exp_val;
            sum_exp += exp_val;
          }

          // Normalize
          for (size_t j = 0; j < seq_len; ++j) {
            size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                              i * seq_len + j;
            attn_weights[attn_idx] /= (sum_exp + 1e-9f);
          }
        }
      }
    }

    // Compute output: O = Attention @ V
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
          for (size_t d = 0; d < head_dim; ++d) {
            T val = 0;
            for (size_t j = 0; j < seq_len; ++j) {
              size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                                i * seq_len + j;
              size_t v_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
              val += attn_weights[attn_idx] * v[v_idx];
            }
            size_t o_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
            output[o_idx] = val;
          }
        }
      }
    }
  } catch (...) {
    delete[] scores;
    delete[] attn_weights;
    throw;
  }

  delete[] scores;
  delete[] attn_weights;
}

template <typename T>
void sdpa_backward(const T *q, const T *k, const T *v, const T *output, const T *grad_output,
                   T *grad_q, T *grad_k, T *grad_v, size_t batch_size, size_t num_heads,
                   size_t seq_len, size_t head_dim, float attn_scale, bool is_causal) {
  // Simplified backward: recompute forward to get attention weights
  // In production, you'd cache these during forward

  const size_t q_stride_b = num_heads * seq_len * head_dim;
  const size_t q_stride_h = seq_len * head_dim;
  const size_t q_stride_s = head_dim;

  T *scores = new T[batch_size * num_heads * seq_len * seq_len];
  T *attn_weights = new T[batch_size * num_heads * seq_len * seq_len];
  T *grad_scores = new T[batch_size * num_heads * seq_len * seq_len];

  try {
    // Recompute forward attention weights
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
          for (size_t j = 0; j < seq_len; ++j) {
            T score = 0;
            for (size_t d = 0; d < head_dim; ++d) {
              size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
              size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
              score += q[q_idx] * k[k_idx];
            }
            score *= static_cast<T>(attn_scale);

            if (is_causal && j > i) {
              score = -std::numeric_limits<T>::infinity();
            }

            size_t score_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                               i * seq_len + j;
            scores[score_idx] = score;
          }
        }

        // Softmax
        for (size_t i = 0; i < seq_len; ++i) {
          T max_score = -std::numeric_limits<T>::infinity();
          for (size_t j = 0; j < seq_len; ++j) {
            size_t score_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                               i * seq_len + j;
            if (std::isfinite(scores[score_idx])) {
              max_score = std::max(max_score, scores[score_idx]);
            }
          }

          T sum_exp = 0;
          for (size_t j = 0; j < seq_len; ++j) {
            size_t score_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                               i * seq_len + j;
            T exp_val =
                std::isfinite(scores[score_idx]) ? std::exp(scores[score_idx] - max_score) : 0;
            size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                              i * seq_len + j;
            attn_weights[attn_idx] = exp_val;
            sum_exp += exp_val;
          }

          for (size_t j = 0; j < seq_len; ++j) {
            size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                              i * seq_len + j;
            attn_weights[attn_idx] /= (sum_exp + 1e-9f);
          }
        }
      }
    }

    // Compute gradients
    // grad_V = A^T @ grad_O
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t j = 0; j < seq_len; ++j) {
          for (size_t d = 0; d < head_dim; ++d) {
            T val = 0;
            for (size_t i = 0; i < seq_len; ++i) {
              size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                                i * seq_len + j;
              size_t grad_o_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
              val += attn_weights[attn_idx] * grad_output[grad_o_idx];
            }
            size_t v_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
            grad_v[v_idx] = val;
          }
        }
      }
    }

    // grad_scores = grad_O @ V^T (element-wise with attention weights for chain rule)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
          for (size_t j = 0; j < seq_len; ++j) {
            T val = 0;
            for (size_t d = 0; d < head_dim; ++d) {
              size_t grad_o_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
              size_t v_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
              val += grad_output[grad_o_idx] * v[v_idx];
            }

            // Apply softmax gradient
            size_t attn_idx = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                              i * seq_len + j;
            T p = attn_weights[attn_idx];

            // Compute sum of grad_scores * attention for numerical stability
            T sum_grad_scores_p = 0;
            for (size_t jj = 0; jj < seq_len; ++jj) {
              size_t attn_idx2 = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len) +
                                 i * seq_len + jj;
              T p2 = attn_weights[attn_idx2];

              T val2 = 0;
              for (size_t d = 0; d < head_dim; ++d) {
                size_t grad_o_idx2 = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
                size_t v_idx2 = b * q_stride_b + h * q_stride_h + jj * q_stride_s + d;
                val2 += grad_output[grad_o_idx2] * v[v_idx2];
              }

              sum_grad_scores_p += p2 * val2;
            }

            size_t grad_scores_idx = b * (num_heads * seq_len * seq_len) +
                                     h * (seq_len * seq_len) + i * seq_len + j;
            grad_scores[grad_scores_idx] = p * (val - sum_grad_scores_p);
          }
        }
      }
    }

    // grad_Q = grad_scores @ K (scaled)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
          for (size_t d = 0; d < head_dim; ++d) {
            T val = 0;
            for (size_t j = 0; j < seq_len; ++j) {
              size_t grad_scores_idx = b * (num_heads * seq_len * seq_len) +
                                       h * (seq_len * seq_len) + i * seq_len + j;
              size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
              val += grad_scores[grad_scores_idx] * k[k_idx];
            }
            val *= static_cast<T>(attn_scale);
            size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
            grad_q[q_idx] = val;
          }
        }
      }
    }

    // grad_K = grad_scores^T @ Q (scaled)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t j = 0; j < seq_len; ++j) {
          for (size_t d = 0; d < head_dim; ++d) {
            T val = 0;
            for (size_t i = 0; i < seq_len; ++i) {
              size_t grad_scores_idx = b * (num_heads * seq_len * seq_len) +
                                       h * (seq_len * seq_len) + i * seq_len + j;
              size_t q_idx = b * q_stride_b + h * q_stride_h + i * q_stride_s + d;
              val += grad_scores[grad_scores_idx] * q[q_idx];
            }
            val *= static_cast<T>(attn_scale);
            size_t k_idx = b * q_stride_b + h * q_stride_h + j * q_stride_s + d;
            grad_k[k_idx] = val;
          }
        }
      }
    }

  } catch (...) {
    delete[] scores;
    delete[] attn_weights;
    delete[] grad_scores;
    throw;
  }

  delete[] scores;
  delete[] attn_weights;
  delete[] grad_scores;
}

// Explicit instantiations
template void sdpa_forward<float>(const float *, const float *, const float *, float *, size_t,
                                  size_t, size_t, size_t, float, bool);
template void sdpa_forward<double>(const double *, const double *, const double *, double *, size_t,
                                   size_t, size_t, size_t, float, bool);

template void sdpa_backward<float>(const float *, const float *, const float *, const float *,
                                   const float *, float *, float *, float *, size_t, size_t, size_t,
                                   size_t, float, bool);
template void sdpa_backward<double>(const double *, const double *, const double *, const double *,
                                    const double *, double *, double *, double *, size_t, size_t,
                                    size_t, size_t, float, bool);

}  // namespace cpu
}  // namespace tnn
