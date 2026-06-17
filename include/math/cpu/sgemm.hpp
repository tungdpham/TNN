/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace synet {
namespace cpu {
void sgemm(const float *A, const float *B, float *C, size_t M, size_t N, size_t K,
           const bool trans_A, const bool trans_B, const float alpha = 1.0f,
           const float beta = 1.0f);

void sgemm_strided(const float *A, const float *B, float *C, size_t M, size_t N, size_t K,
                   const bool trans_A, const bool trans_B, const float alpha, const float beta,
                   size_t lda, size_t ldb, size_t ldc);

}  // namespace cpu
}  // namespace synet