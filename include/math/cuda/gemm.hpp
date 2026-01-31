/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void gemm_ex(const A_T *A, const B_T *B, C_T *C, const size_t M, const size_t N, const size_t K,
             const bool transpose_A, const bool transpose_B, const Compute_T alpha,
             const Compute_T beta, const size_t lda, const size_t ldb, const size_t ldc,
             cudaStream_t stream);

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void gemm_strided_batched_ex(const A_T *A, const B_T *B, C_T *C, const size_t M, const size_t N,
                             const size_t K, const bool transpose_A, const bool transpose_B,
                             const Compute_T alpha, const Compute_T beta, const size_t lda,
                             const size_t ldb, const size_t ldc, const size_t strideA,
                             const size_t strideB, const size_t strideC, const size_t batch_count,
                             cudaStream_t stream);

}  // namespace cuda

}  // namespace tnn
#endif