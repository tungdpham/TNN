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
#include <string>

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
} // anonymous namespace

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
inline void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n,
                 const MKL_INT k, const float alpha, const float *a, const MKL_INT lda,
                 const float *b, const MKL_INT ldb, const float beta, float *c, const MKL_INT ldc) {
  cblas_sgemm(CblasRowMajor, transa == 'T' ? CblasTrans : CblasNoTrans,
              transb == 'T' ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c,
              ldc);
}

/**
 * @brief Wrapper for Intel MKL DGEMM (double precision)
 */
inline void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n,
                 const MKL_INT k, const double alpha, const double *a, const MKL_INT lda,
                 const double *b, const MKL_INT ldb, const double beta, double *c,
                 const MKL_INT ldc) {
  cblas_dgemm(CblasRowMajor, transa == 'T' ? CblasTrans : CblasNoTrans,
              transb == 'T' ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c,
              ldc);
}

/**
 * @brief Optimized GEMM for convolution forward pass
 * Computes output = weights * im2col_data
 *
 * @param weights Weight matrix [out_channels x kernel_size]
 * @param im2col_data Im2col matrix [kernel_size x output_size]
 * @param output Output matrix [out_channels x output_size]
 * @param out_channels Number of output channels
 * @param kernel_size Size of convolution kernel (in_channels * kernel_h * kernel_w)
 * @param output_size Size of output spatial dimensions (batch_size * output_h * output_w)
 */
template <typename T>
inline void conv_forward_gemm(const T *weights, const T *im2col_data, T *output,
                              const MKL_INT out_channels, const MKL_INT kernel_size,
                              const MKL_INT output_size) {
  // Compute: output = weights * im2col_data
  // weights: [out_channels x kernel_size]
  // im2col_data: [kernel_size x output_size]
  // output: [out_channels x output_size]
  gemm('N', 'N', out_channels, output_size, kernel_size, T(1.0), weights, kernel_size, im2col_data,
       output_size, T(0.0), output, output_size);
}

/**
 * @brief Optimized GEMM for weight gradient computation
 * Computes weight_grad += output_grad * im2col_data^T
 *
 * @param output_grad Output gradient [out_channels x output_size]
 * @param im2col_data Im2col matrix [kernel_size x output_size]
 * @param weight_grad Weight gradients [out_channels x kernel_size]
 * @param out_channels Number of output channels
 * @param kernel_size Size of convolution kernel
 * @param output_size Size of output spatial dimensions
 */
template <typename T>
inline void conv_weight_grad_gemm(const T *output_grad, const T *im2col_data, T *weight_grad,
                                  const MKL_INT out_channels, const MKL_INT kernel_size,
                                  const MKL_INT output_size) {
  // Compute: weight_grad += output_grad * im2col_data^T
  // output_grad: [out_channels x output_size]
  // im2col_data: [kernel_size x output_size]
  // weight_grad: [out_channels x kernel_size]
  gemm('N', 'T', out_channels, kernel_size, output_size, T(1.0), output_grad, output_size,
       im2col_data, output_size, T(1.0), weight_grad, kernel_size);
}

/**
 * @brief Optimized GEMM for input gradient computation
 * Computes col_grad = weights^T * output_grad
 *
 * @param weights Weight matrix [out_channels x kernel_size]
 * @param output_grad Output gradient [out_channels x output_size]
 * @param col_grad Column gradient [kernel_size x output_size]
 * @param out_channels Number of output channels
 * @param kernel_size Size of convolution kernel
 * @param output_size Size of output spatial dimensions
 */
template <typename T>
inline void conv_input_grad_gemm(const T *weights, const T *output_grad, T *col_grad,
                                 const MKL_INT out_channels, const MKL_INT kernel_size,
                                 const MKL_INT output_size) {
  // Compute: col_grad = weights^T * output_grad
  // weights: [out_channels x kernel_size]
  // output_grad: [out_channels x output_size]
  // col_grad: [kernel_size x output_size]
  gemm('T', 'N', kernel_size, output_size, out_channels, T(1.0), weights, kernel_size, output_grad,
       output_size, T(0.0), col_grad, output_size);
}

} // namespace mkl
} // namespace tnn

#endif // USE_MKL