/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

/**
 * This file provides a wrapper header for gemm functions
 */
#include "dgemm.hpp"
#include "sgemm.hpp"

namespace tmath {
template <typename T>
void gemm(const T *A, const T *B, T *C, const size_t M, const size_t N, const size_t K,
          const bool trans_A = false, const bool trans_B = false) {
  if constexpr (std::is_same<T, float>::value) {
    sgemm(A, B, C, M, N, K, trans_A, trans_B);
  } else if constexpr (std::is_same<T, double>::value) {
    dgemm(A, B, C, M, N, K, trans_A, trans_B);
  } else {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "Unsupported data type for gemm. Only float and double are supported.");
  }
}
} // namespace tmath