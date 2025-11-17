#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
void dgemm(const double *A, const double *B, double *C, const size_t M, const size_t N,
           const size_t K, const bool trans_A, const bool trans_B, const double alpha,
           const double beta);
} // namespace cpu
} // namespace tnn