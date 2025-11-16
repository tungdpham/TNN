/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cuda {

/**
 * @brief CUDA-accelerated General Matrix Multiplication (GEMM)
 *
 * Performs scaled matrix multiplication: C = alpha * A * B (or their transposes)
 * All matrices are assumed to be in row-major order.
 *
 * @tparam T Data type (float or double)
 * @param A Pointer to matrix A on device memory
 * @param B Pointer to matrix B on device memory
 * @param C Pointer to result matrix C on device memory
 * @param M Number of rows in A (or A^T if trans_A is true)
 * @param N Number of columns in B (or B^T if trans_B is true)
 * @param K Number of columns in A / rows in B (or vice versa with transpose)
 * @param alpha Scalar multiplier for the matrix product
 * @param trans_A Whether to transpose A
 * @param trans_B Whether to transpose B
 */
template <typename T>
void gemm(const T *A, const T *B, T *C, const size_t M, const size_t N, const size_t K,
          const T alpha = 1.0, const bool trans_A = false, const bool trans_B = false);

} // namespace cuda

} // namespace tnn