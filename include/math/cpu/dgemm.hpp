#pragma once

#include <cstddef>

namespace tunx {
namespace cpu {
void dgemm(const double *A, const double *B, double *C, size_t M, size_t N, size_t K,
           const bool trans_A, const bool trans_B, const double alpha, const double beta);
}  // namespace cpu
}  // namespace tunx