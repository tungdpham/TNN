#pragma once

#ifdef USE_MKL
#include "utils/mkl_utils.hpp"
#endif
#include "dgemm.hpp"
#include "sgemm.hpp"
#include <type_traits>

namespace tnn {
namespace cpu {

template <typename T>
void gemm(const T *A, const T *B, T *C, const size_t M, const size_t N, const size_t K,
          const bool trans_A, const bool trans_B, const T alpha, const T beta) {
#ifdef USE_MKL
  char transa = trans_A ? 'T' : 'N';
  char transb = trans_B ? 'T' : 'N';
  mkl::gemm(transa, transb, static_cast<MKL_INT>(M), static_cast<MKL_INT>(N),
            static_cast<MKL_INT>(K), alpha, A, static_cast<MKL_INT>(trans_A ? M : K), B,
            static_cast<MKL_INT>(trans_B ? K : N), beta, C, static_cast<MKL_INT>(N));
#else
  if constexpr (std::is_same<T, float>::value) {
    sgemm(A, B, C, M, N, K, trans_A, trans_B, alpha, beta);
  } else if constexpr (std::is_same<T, double>::value) {
    dgemm(A, B, C, M, N, K, trans_A, trans_B, alpha, beta);
  } else {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "Unsupported data type for gemm. Only float and double are supported.");
  }
#endif
}

} // namespace cpu
} // namespace tnn