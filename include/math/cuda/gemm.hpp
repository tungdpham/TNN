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

namespace synet {
namespace cuda {

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void gemm_ex(const A_T *A, const B_T *B, C_T *C, size_t M, size_t N, size_t K,
             const bool transpose_A, const bool transpose_B, const Compute_T alpha,
             const Compute_T beta, size_t lda, size_t ldb, size_t ldc, cudaStream_t stream);

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void gemm_strided_batched_ex(const A_T *A, const B_T *B, C_T *C, size_t M, size_t N, size_t K,
                             const bool transpose_A, const bool transpose_B, const Compute_T alpha,
                             const Compute_T beta, size_t lda, size_t ldb, size_t ldc,
                             size_t strideA, size_t strideB, size_t strideC, size_t batch_count,
                             cudaStream_t stream);

}  // namespace cuda

}  // namespace synet
#endif