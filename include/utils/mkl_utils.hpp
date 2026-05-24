/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_MKL

#include <mkl.h>

#include <stdexcept>
#include <type_traits>

namespace tnn {
namespace mkl {

inline void initialize_mkl() {
#ifdef USE_TBB
  mkl_set_threading_layer(MKL_THREADING_TBB);
#elif defined(USE_OMP)
  mkl_set_threading_layer(MKL_THREADING_OMP);
#endif
}

// Global constructor to automatically initialize MKL
namespace {
struct MklInitializer {
  MklInitializer() { initialize_mkl(); }
};
static MklInitializer mkl_init;
}  // anonymous namespace

/**
 * @brief Wrapper for Intel MKL SGEMM (single precision)
 * Performs C = alpha * A * B + beta * C
 *
 * @param transa Whether to transpose matrix A ('N' for no transpose, 'T' for transpose)
 * @param transb Whether to transpose matrix B ('N' for no transpose, 'T' for transpose)
 * @param m Number of rows in A (and C)
 * @param n Number of columns in B (and C)
 * @param k Number of columns in A (and rows in B)
 * @param alpha Scalar multiplier for A*B
 * @param a Pointer to matrix A
 * @param lda Leading dimension of A
 * @param b Pointer to matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param c Pointer to matrix C (output)
 * @param ldc Leading dimension of C
 */
template <typename T>
inline void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n,
                 const MKL_INT k, const T alpha, const T *a, const MKL_INT lda, const T *b,
                 const MKL_INT ldb, const T beta, T *c, const MKL_INT ldc) {
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, transa == 'T' ? CblasTrans : CblasNoTrans,
                transb == 'T' ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c,
                ldc);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, transa == 'T' ? CblasTrans : CblasNoTrans,
                transb == 'T' ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c,
                ldc);
  } else {
    throw std::runtime_error("Unsupported data type for MKL GEMM");
  }
}

}  // namespace mkl
}  // namespace tnn

#endif  // USE_MKL