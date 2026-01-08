#pragma once

#ifdef USE_MKL
#include "utils/mkl_utils.hpp"
#endif
#include "dgemm.hpp"
#include "sgemm.hpp"
#include <type_traits>
#include <vector>

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

template <typename T>
void gemm_strided_batched(const T *A, const T *B, T *C, const size_t M, const size_t N,
                          const size_t K, const bool trans_A, const bool trans_B, const T alpha,
                          const T beta, const size_t batch_count, const size_t stride_A,
                          const size_t stride_B, const size_t stride_C) {
#ifdef USE_MKL
  std::vector<const T *> a_ptrs(batch_count);
  std::vector<const T *> b_ptrs(batch_count);
  std::vector<T *> c_ptrs(batch_count);
  for (size_t i = 0; i < batch_count; ++i) {
    a_ptrs[i] = A + i * stride_A;
    b_ptrs[i] = B + i * stride_B;
    c_ptrs[i] = C + i * stride_C;
  }
  char transa = trans_A ? 'T' : 'N';
  char transb = trans_B ? 'T' : 'N';
  mkl::gemm_batch(transa, transb, static_cast<MKL_INT>(M), static_cast<MKL_INT>(N),
                  static_cast<MKL_INT>(K), alpha, a_ptrs.data(),
                  static_cast<MKL_INT>(trans_A ? M : K), b_ptrs.data(),
                  static_cast<MKL_INT>(trans_B ? K : N), beta, c_ptrs.data(),
                  static_cast<MKL_INT>(N), static_cast<MKL_INT>(batch_count));
#else
  for (size_t i = 0; i < batch_count; ++i) {
    const T *curr_A = A + i * stride_A;
    const T *curr_B = B + i * stride_B;
    T *curr_C = C + i * stride_C;
    gemm(curr_A, curr_B, curr_C, M, N, K, trans_A, trans_B, alpha, beta);
  }
#endif
}

} // namespace cpu
} // namespace tnn