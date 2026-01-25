#pragma once

#ifdef USE_MKL
#include "utils/mkl_utils.hpp"
#else
#include "dgemm.hpp"
#include "sgemm.hpp"
#endif
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tnn {
namespace cpu {

template <typename T>
void gemm(const T *A, const T *B, T *C, const size_t M, const size_t N, const size_t K,
          const bool trans_A, const bool trans_B, const T alpha, const T beta) {
#ifdef USE_MKL
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    char transa = trans_A ? 'T' : 'N';
    char transb = trans_B ? 'T' : 'N';
    mkl::gemm(transa, transb, static_cast<MKL_INT>(M), static_cast<MKL_INT>(N),
              static_cast<MKL_INT>(K), alpha, A, static_cast<MKL_INT>(trans_A ? M : K), B,
              static_cast<MKL_INT>(trans_B ? K : N), beta, C, static_cast<MKL_INT>(N));
  } else {
    throw std::runtime_error("Unsupported data type for GEMM on CPU/MKL");
  }
#else
  if constexpr (std::is_same<T, float>::value) {
    sgemm(A, B, C, M, N, K, trans_A, trans_B, alpha, beta);
  } else if constexpr (std::is_same<T, double>::value) {
    dgemm(A, B, C, M, N, K, trans_A, trans_B, alpha, beta);
  } else {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        T sum = static_cast<T>(0);
        for (size_t k = 0; k < K; ++k) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = alpha * sum + beta * C[i * N + j];
      }
    }
  }
#endif
}

} // namespace cpu
} // namespace tnn