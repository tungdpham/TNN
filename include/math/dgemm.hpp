#pragma once

#include <cstddef>

namespace tmath {
void dgemm(const double *A, const double *B, double *C, const size_t M, const size_t N,
           const size_t K, const bool trans_A = false, const bool trans_B = false);
} // namespace tmath