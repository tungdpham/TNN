/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "utils/ops.hpp"
#include "utils/parallel_for.hpp"
#include "utils/simd_asm.hpp"
#include <algorithm>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <cstdint>
#include <cstdlib>
#include <memory>

namespace tmath {
void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K,
           const bool trans_A = false, const bool trans_B = false);
void dgemm(const double *A, const double *B, double *C, const int M, const int N, const int K);

} // namespace tmath