/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "math/cpu/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace tnn {
namespace cpu {

template <typename T>
void flash_attention_forward(T *q, T *k, T *v, T *output, size_t batch_count, size_t head_dim,
                             size_t seq_len, bool is_causal = true) {
  const size_t Br = 64;
  const size_t Bc = 64;
  size_t D = head_dim;
  size_t L = seq_len;

  for (size_t b = 0; b < batch_count; ++b) {
    T *Q_b = q + b * D * L;
    T *K_b = k + b * D * L;
    T *V_b = v + b * D * L;
    T *O_b = output + b * D * L;

    std::fill(O_b, O_b + D * L, static_cast(0));

    std::vector m(L, -INFINITY);
    std::vector l(L, 0);

    // Pre-allocate buffers for blocks
    std::vector Q_block(D * Br);
    std::vector K_block(D * Bc);
    std::vector V_block(D * Bc);
    std::vector S_ij(Br * Bc);
    std::vector P_ij(Br * Bc);
    std::vector PV(D * Br);
    std::vector m_block(Br);
    std::vector l_block(Br);

    for (size_t i = 0; i < L; i += Br) {
      size_t br = std::min(Br, L - i);

      for (size_t j = 0; j < L; j += Bc) {
        size_t bc = std::min(Bc, L - j);

        // Copy blocks to contiguous memory
        for (size_t d = 0; d < D; ++d) {
          for (size_t r = 0; r < br; ++r) {
            Q_block[d * br + r] = Q_b[d * L + (i + r)];
          }
        }

        for (size_t d = 0; d < D; ++d) {
          for (size_t c = 0; c < bc; ++c) {
            K_block[d * bc + c] = K_b[d * L + (j + c)];
          }
        }

        T scale = 1.0f / std::sqrt(static_cast(D));
        cpu::gemm(Q_block->data_as(), K_block->data_as(), S_ij->data_as(), br, bc, D, true, false,
                  scale, 0.0f);

        if (is_causal) {
          for (size_t r = 0; r < br; ++r) {
            size_t global_r = i + r;
            for (size_t c = 0; c < bc; ++c) {
              size_t global_c = j + c;
              if (global_c > global_r) {
                S_ij[r * bc + c] = -INFINITY;
              }
            }
          }
        }

        std::fill(m_block.begin(), m_block.end(), -INFINITY);

        for (size_t r = 0; r < br; ++r) {
          T max_val = -INFINITY;
          for (size_t c = 0; c < bc; ++c) {
            if (S_ij[r * bc + c] > max_val)
              max_val = S_ij[r * bc + c];
          }
          m_block[r] = max_val;
        }

        for (size_t r = 0; r < br; ++r) {
          T sum_val = 0;
          for (size_t c = 0; c < bc; ++c) {
            T val = std::exp(S_ij[r * bc + c] - m_block[r]);
            P_ij[r * bc + c] = val;
            sum_val += val;
          }
          l_block[r] = sum_val;
        }

        for (size_t d = 0; d < D; ++d) {
          for (size_t c = 0; c < bc; ++c) {
            V_block[d * bc + c] = V_b[d * L + (j + c)];
          }
        }

        cpu::gemm(V_block->data_as(), P_ij->data_as(), PV->data_as(), D, br, bc, false, true, 1.0f,
                  0.0f);

        for (size_t r = 0; r < br; ++r) {
          size_t global_r = i + r;
          T m_prev = m[global_r];
          T l_prev = l[global_r];
          T m_curr = m_block[r];
          T l_curr = l_block[r];

          T m_new = std::max(m_prev, m_curr);
          T l_new = std::exp(m_prev - m_new) * l_prev + std::exp(m_curr - m_new) * l_curr;

          T factor_prev = std::exp(m_prev - m_new);
          T factor_curr = std::exp(m_curr - m_new);

          for (size_t d = 0; d < D; ++d) {
            T &o_val = O_b[d * L + global_r];
            T pv_val = PV[d * br + r];
            o_val = factor_prev * o_val + factor_curr * pv_val;
          }

          m[global_r] = m_new;
          l[global_r] = l_new;
        }
      }

      for (size_t r = 0; r < br; ++r) {
        size_t global_r = i + r;
        T inv_l = 1.0f / l[global_r];
        for (size_t d = 0; d < D; ++d) {
          O_b[d * L + global_r] *= inv_l;
        }
      }
    }
  }
}

} // namespace cpu
} // namespace tnn
