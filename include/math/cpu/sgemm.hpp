/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
void sgemm(const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K,
           const bool trans_A = false, const bool trans_B = false);
} // namespace cpu
} // namespace tnn