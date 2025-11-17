#include "math/cpu/dgemm.hpp"
#include "threading/thread_handler.hpp"
#include <algorithm> // For std::min
#include <cstddef>
#include <cstring>

namespace tnn {
namespace cpu {

constexpr size_t DEFAULT_BLOCK_SIZE = 32;

#ifdef __AVX2__
inline bool is_aligned_32(const void *ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0; }

inline void dgemm_kernel_avx2_nn(const double *A, const double *B, double *C, const size_t M,
                                 const size_t N, const size_t K, const size_t i, const size_t j,
                                 const size_t k, const size_t i_max, const size_t j_max,
                                 const size_t k_max, const double alpha) {
  __m256d alpha_vec = _mm256_broadcast_sd(&alpha);
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec_0 = _mm256_setzero_pd();
      __m256d c_vec_1 = _mm256_setzero_pd();
      __m256d c_vec_2 = _mm256_setzero_pd();
      __m256d c_vec_3 = _mm256_setzero_pd();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d b_vec = _mm256_loadu_pd(&B[kk * N + jj]);
        __m256d a_vec_0 = _mm256_broadcast_sd(&A[(ii + 0) * K + kk]);
        __m256d a_vec_1 = _mm256_broadcast_sd(&A[(ii + 1) * K + kk]);
        __m256d a_vec_2 = _mm256_broadcast_sd(&A[(ii + 2) * K + kk]);
        __m256d a_vec_3 = _mm256_broadcast_sd(&A[(ii + 3) * K + kk]);
        c_vec_0 = _mm256_fmadd_pd(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_pd(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_pd(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_pd(a_vec_3, b_vec, c_vec_3);
      }
      __m256d c_old_0 = _mm256_loadu_pd(&C[(ii + 0) * N + jj]);
      __m256d c_old_1 = _mm256_loadu_pd(&C[(ii + 1) * N + jj]);
      __m256d c_old_2 = _mm256_loadu_pd(&C[(ii + 2) * N + jj]);
      __m256d c_old_3 = _mm256_loadu_pd(&C[(ii + 3) * N + jj]);
      _mm256_storeu_pd(&C[(ii + 0) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_0, c_old_0));
      _mm256_storeu_pd(&C[(ii + 1) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_1, c_old_1));
      _mm256_storeu_pd(&C[(ii + 2) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_2, c_old_2));
      _mm256_storeu_pd(&C[(ii + 3) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_3, c_old_3));
    }
    for (; jj < j_max; ++jj) {
      double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        double b_val = B[kk * N + jj];
        sum0 += A[(ii + 0) * K + kk] * b_val;
        sum1 += A[(ii + 1) * K + kk] * b_val;
        sum2 += A[(ii + 2) * K + kk] * b_val;
        sum3 += A[(ii + 3) * K + kk] * b_val;
      }
      C[(ii + 0) * N + jj] += alpha * sum0;
      C[(ii + 1) * N + jj] += alpha * sum1;
      C[(ii + 2) * N + jj] += alpha * sum2;
      C[(ii + 3) * N + jj] += alpha * sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec = _mm256_setzero_pd();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d a_vec = _mm256_broadcast_sd(&A[ii * K + kk]);
        __m256d b_vec = _mm256_loadu_pd(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
      }
      __m256d c_old = _mm256_loadu_pd(&C[ii * N + jj]);
      _mm256_storeu_pd(&C[ii * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec, c_old));
    }
    for (; jj < j_max; ++jj) {
      double sum = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += alpha * sum;
    }
  }
}

inline void dgemm_kernel_avx2_nn_aligned(const double *A, const double *B, double *C,
                                         const size_t M, const size_t N, const size_t K,
                                         const size_t i, const size_t j, const size_t k,
                                         const size_t i_max, const size_t j_max, const size_t k_max,
                                         const double alpha) {
  __m256d alpha_vec = _mm256_broadcast_sd(&alpha);
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec_0 = _mm256_setzero_pd();
      __m256d c_vec_1 = _mm256_setzero_pd();
      __m256d c_vec_2 = _mm256_setzero_pd();
      __m256d c_vec_3 = _mm256_setzero_pd();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d b_vec = _mm256_load_pd(&B[kk * N + jj]);
        __m256d a_vec_0 = _mm256_broadcast_sd(&A[(ii + 0) * K + kk]);
        __m256d a_vec_1 = _mm256_broadcast_sd(&A[(ii + 1) * K + kk]);
        __m256d a_vec_2 = _mm256_broadcast_sd(&A[(ii + 2) * K + kk]);
        __m256d a_vec_3 = _mm256_broadcast_sd(&A[(ii + 3) * K + kk]);
        c_vec_0 = _mm256_fmadd_pd(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_pd(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_pd(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_pd(a_vec_3, b_vec, c_vec_3);
      }
      __m256d c_old_0 = _mm256_load_pd(&C[(ii + 0) * N + jj]);
      __m256d c_old_1 = _mm256_load_pd(&C[(ii + 1) * N + jj]);
      __m256d c_old_2 = _mm256_load_pd(&C[(ii + 2) * N + jj]);
      __m256d c_old_3 = _mm256_load_pd(&C[(ii + 3) * N + jj]);
      _mm256_store_pd(&C[(ii + 0) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_0, c_old_0));
      _mm256_store_pd(&C[(ii + 1) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_1, c_old_1));
      _mm256_store_pd(&C[(ii + 2) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_2, c_old_2));
      _mm256_store_pd(&C[(ii + 3) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_3, c_old_3));
    }
    for (; jj < j_max; ++jj) {
      double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        double b_val = B[kk * N + jj];
        sum0 += A[(ii + 0) * K + kk] * b_val;
        sum1 += A[(ii + 1) * K + kk] * b_val;
        sum2 += A[(ii + 2) * K + kk] * b_val;
        sum3 += A[(ii + 3) * K + kk] * b_val;
      }
      C[(ii + 0) * N + jj] += alpha * sum0;
      C[(ii + 1) * N + jj] += alpha * sum1;
      C[(ii + 2) * N + jj] += alpha * sum2;
      C[(ii + 3) * N + jj] += alpha * sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec = _mm256_setzero_pd();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d a_vec = _mm256_broadcast_sd(&A[ii * K + kk]);
        __m256d b_vec = _mm256_load_pd(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
      }
      __m256d c_old = _mm256_load_pd(&C[ii * N + jj]);
      _mm256_store_pd(&C[ii * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec, c_old));
    }
    for (; jj < j_max; ++jj) {
      double sum = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += alpha * sum;
    }
  }
}

inline void dgemm_kernel_avx2_nt(const double *A, const double *B, double *C, const size_t M,
                                 const size_t N, const size_t K, const size_t i, const size_t j,
                                 const size_t k, const size_t i_max, const size_t j_max,
                                 const size_t k_max, const double alpha) {
  for (size_t ii = i; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d sum0 = _mm256_setzero_pd();
      __m256d sum1 = _mm256_setzero_pd();
      __m256d sum2 = _mm256_setzero_pd();
      __m256d sum3 = _mm256_setzero_pd();

      size_t kk = k;
      for (; kk + 3 < k_max; kk += 4) {
        __m256d a_vec = _mm256_loadu_pd(&A[ii * K + kk]);
        __m256d b0_vec = _mm256_loadu_pd(&B[(jj + 0) * K + kk]);
        __m256d b1_vec = _mm256_loadu_pd(&B[(jj + 1) * K + kk]);
        __m256d b2_vec = _mm256_loadu_pd(&B[(jj + 2) * K + kk]);
        __m256d b3_vec = _mm256_loadu_pd(&B[(jj + 3) * K + kk]);

        sum0 = _mm256_fmadd_pd(a_vec, b0_vec, sum0);
        sum1 = _mm256_fmadd_pd(a_vec, b1_vec, sum1);
        sum2 = _mm256_fmadd_pd(a_vec, b2_vec, sum2);
        sum3 = _mm256_fmadd_pd(a_vec, b3_vec, sum3);
      }

      auto horizontal_sum = [](const __m256d &vec) -> double {
        __m128d vlow = _mm256_castpd256_pd128(vec);
        __m128d vhigh = _mm256_extractf128_pd(vec, 1);
        vlow = _mm_add_pd(vlow, vhigh);
        vlow = _mm_hadd_pd(vlow, vlow);
        return _mm_cvtsd_f64(vlow);
      };

      double partial_sum0 = horizontal_sum(sum0);
      double partial_sum1 = horizontal_sum(sum1);
      double partial_sum2 = horizontal_sum(sum2);
      double partial_sum3 = horizontal_sum(sum3);

      for (; kk < k_max; ++kk) {
        double a_val = A[ii * K + kk];
        partial_sum0 += a_val * B[(jj + 0) * K + kk];
        partial_sum1 += a_val * B[(jj + 1) * K + kk];
        partial_sum2 += a_val * B[(jj + 2) * K + kk];
        partial_sum3 += a_val * B[(jj + 3) * K + kk];
      }

      C[ii * N + jj + 0] += alpha * partial_sum0;
      C[ii * N + jj + 1] += alpha * partial_sum1;
      C[ii * N + jj + 2] += alpha * partial_sum2;
      C[ii * N + jj + 3] += alpha * partial_sum3;
    }

    for (; jj < j_max; ++jj) {
      __m256d sum_vec = _mm256_setzero_pd();
      size_t kk = k;
      for (; kk + 3 < k_max; kk += 4) {
        __m256d a_vec = _mm256_loadu_pd(&A[ii * K + kk]);
        __m256d b_vec = _mm256_loadu_pd(&B[jj * K + kk]);
        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
      }

      __m128d vlow = _mm256_castpd256_pd128(sum_vec);
      __m128d vhigh = _mm256_extractf128_pd(sum_vec, 1);
      vlow = _mm_add_pd(vlow, vhigh);
      vlow = _mm_hadd_pd(vlow, vlow);
      double sum = _mm_cvtsd_f64(vlow);

      for (; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[jj * K + kk];
      }

      C[ii * N + jj] += alpha * sum;
    }
  }
}

inline void dgemm_kernel_avx2_nt_aligned(const double *A, const double *B, double *C,
                                         const size_t M, const size_t N, const size_t K,
                                         const size_t i, const size_t j, const size_t k,
                                         const size_t i_max, const size_t j_max, const size_t k_max,
                                         const double alpha) {
  for (size_t ii = i; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d sum0 = _mm256_setzero_pd();
      __m256d sum1 = _mm256_setzero_pd();
      __m256d sum2 = _mm256_setzero_pd();
      __m256d sum3 = _mm256_setzero_pd();

      size_t kk = k;
      for (; kk + 3 < k_max; kk += 4) {
        __m256d a_vec = _mm256_load_pd(&A[ii * K + kk]);
        __m256d b0_vec = _mm256_load_pd(&B[(jj + 0) * K + kk]);
        __m256d b1_vec = _mm256_load_pd(&B[(jj + 1) * K + kk]);
        __m256d b2_vec = _mm256_load_pd(&B[(jj + 2) * K + kk]);
        __m256d b3_vec = _mm256_load_pd(&B[(jj + 3) * K + kk]);

        sum0 = _mm256_fmadd_pd(a_vec, b0_vec, sum0);
        sum1 = _mm256_fmadd_pd(a_vec, b1_vec, sum1);
        sum2 = _mm256_fmadd_pd(a_vec, b2_vec, sum2);
        sum3 = _mm256_fmadd_pd(a_vec, b3_vec, sum3);
      }

      auto horizontal_sum = [](const __m256d &vec) -> double {
        __m128d vlow = _mm256_castpd256_pd128(vec);
        __m128d vhigh = _mm256_extractf128_pd(vec, 1);
        vlow = _mm_add_pd(vlow, vhigh);
        vlow = _mm_hadd_pd(vlow, vlow);
        return _mm_cvtsd_f64(vlow);
      };

      double partial_sum0 = horizontal_sum(sum0);
      double partial_sum1 = horizontal_sum(sum1);
      double partial_sum2 = horizontal_sum(sum2);
      double partial_sum3 = horizontal_sum(sum3);

      for (; kk < k_max; ++kk) {
        double a_val = A[ii * K + kk];
        partial_sum0 += a_val * B[(jj + 0) * K + kk];
        partial_sum1 += a_val * B[(jj + 1) * K + kk];
        partial_sum2 += a_val * B[(jj + 2) * K + kk];
        partial_sum3 += a_val * B[(jj + 3) * K + kk];
      }

      C[ii * N + jj + 0] += alpha * partial_sum0;
      C[ii * N + jj + 1] += alpha * partial_sum1;
      C[ii * N + jj + 2] += alpha * partial_sum2;
      C[ii * N + jj + 3] += alpha * partial_sum3;
    }

    for (; jj < j_max; ++jj) {
      __m256d sum_vec = _mm256_setzero_pd();
      size_t kk = k;
      for (; kk + 3 < k_max; kk += 4) {
        __m256d a_vec = _mm256_load_pd(&A[ii * K + kk]);
        __m256d b_vec = _mm256_load_pd(&B[jj * K + kk]);
        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
      }

      __m128d vlow = _mm256_castpd256_pd128(sum_vec);
      __m128d vhigh = _mm256_extractf128_pd(sum_vec, 1);
      vlow = _mm_add_pd(vlow, vhigh);
      vlow = _mm_hadd_pd(vlow, vlow);
      double sum = _mm_cvtsd_f64(vlow);

      for (; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[jj * K + kk];
      }

      C[ii * N + jj] += alpha * sum;
    }
  }
}

inline void dgemm_kernel_avx2_tn(const double *A, const double *B, double *C, const size_t M,
                                 const size_t N, const size_t K, const size_t i, const size_t j,
                                 const size_t k, const size_t i_max, const size_t j_max,
                                 const size_t k_max, const double alpha) {
  __m256d alpha_vec = _mm256_broadcast_sd(&alpha);
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec_0 = _mm256_setzero_pd();
      __m256d c_vec_1 = _mm256_setzero_pd();
      __m256d c_vec_2 = _mm256_setzero_pd();
      __m256d c_vec_3 = _mm256_setzero_pd();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d b_vec = _mm256_loadu_pd(&B[kk * N + jj]);
        __m256d a_vec_0 = _mm256_broadcast_sd(&A[kk * M + (ii + 0)]);
        __m256d a_vec_1 = _mm256_broadcast_sd(&A[kk * M + (ii + 1)]);
        __m256d a_vec_2 = _mm256_broadcast_sd(&A[kk * M + (ii + 2)]);
        __m256d a_vec_3 = _mm256_broadcast_sd(&A[kk * M + (ii + 3)]);
        c_vec_0 = _mm256_fmadd_pd(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_pd(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_pd(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_pd(a_vec_3, b_vec, c_vec_3);
      }

      __m256d c_old_0 = _mm256_loadu_pd(&C[(ii + 0) * N + jj]);
      __m256d c_old_1 = _mm256_loadu_pd(&C[(ii + 1) * N + jj]);
      __m256d c_old_2 = _mm256_loadu_pd(&C[(ii + 2) * N + jj]);
      __m256d c_old_3 = _mm256_loadu_pd(&C[(ii + 3) * N + jj]);
      _mm256_storeu_pd(&C[(ii + 0) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_0, c_old_0));
      _mm256_storeu_pd(&C[(ii + 1) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_1, c_old_1));
      _mm256_storeu_pd(&C[(ii + 2) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_2, c_old_2));
      _mm256_storeu_pd(&C[(ii + 3) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_3, c_old_3));
    }
    for (; jj < j_max; ++jj) {
      double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        double b_val = B[kk * N + jj];
        sum0 += A[kk * M + (ii + 0)] * b_val;
        sum1 += A[kk * M + (ii + 1)] * b_val;
        sum2 += A[kk * M + (ii + 2)] * b_val;
        sum3 += A[kk * M + (ii + 3)] * b_val;
      }
      C[(ii + 0) * N + jj] += alpha * sum0;
      C[(ii + 1) * N + jj] += alpha * sum1;
      C[(ii + 2) * N + jj] += alpha * sum2;
      C[(ii + 3) * N + jj] += alpha * sum3;
    }
  }

  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec = _mm256_setzero_pd();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d a_vec = _mm256_broadcast_sd(&A[kk * M + ii]);
        __m256d b_vec = _mm256_loadu_pd(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
      }
      __m256d c_old = _mm256_loadu_pd(&C[ii * N + jj]);
      _mm256_storeu_pd(&C[ii * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec, c_old));
    }

    for (; jj < j_max; ++jj) {
      double sum = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[kk * M + ii] * B[kk * N + jj];
      }
      C[ii * N + jj] += alpha * sum;
    }
  }
}

inline void dgemm_kernel_avx2_tn_aligned(const double *A, const double *B, double *C,
                                         const size_t M, const size_t N, const size_t K,
                                         const size_t i, const size_t j, const size_t k,
                                         const size_t i_max, const size_t j_max, const size_t k_max,
                                         const double alpha) {
  __m256d alpha_vec = _mm256_broadcast_sd(&alpha);
  size_t ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec_0 = _mm256_setzero_pd();
      __m256d c_vec_1 = _mm256_setzero_pd();
      __m256d c_vec_2 = _mm256_setzero_pd();
      __m256d c_vec_3 = _mm256_setzero_pd();

      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d b_vec = _mm256_load_pd(&B[kk * N + jj]);
        __m256d a_vec_0 = _mm256_broadcast_sd(&A[kk * M + (ii + 0)]);
        __m256d a_vec_1 = _mm256_broadcast_sd(&A[kk * M + (ii + 1)]);
        __m256d a_vec_2 = _mm256_broadcast_sd(&A[kk * M + (ii + 2)]);
        __m256d a_vec_3 = _mm256_broadcast_sd(&A[kk * M + (ii + 3)]);
        c_vec_0 = _mm256_fmadd_pd(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_pd(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_pd(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_pd(a_vec_3, b_vec, c_vec_3);
      }
      __m256d c_old_0 = _mm256_load_pd(&C[(ii + 0) * N + jj]);
      __m256d c_old_1 = _mm256_load_pd(&C[(ii + 1) * N + jj]);
      __m256d c_old_2 = _mm256_load_pd(&C[(ii + 2) * N + jj]);
      __m256d c_old_3 = _mm256_load_pd(&C[(ii + 3) * N + jj]);
      _mm256_store_pd(&C[(ii + 0) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_0, c_old_0));
      _mm256_store_pd(&C[(ii + 1) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_1, c_old_1));
      _mm256_store_pd(&C[(ii + 2) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_2, c_old_2));
      _mm256_store_pd(&C[(ii + 3) * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec_3, c_old_3));
    }
    for (; jj < j_max; ++jj) {
      double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        double b_val = B[kk * N + jj];
        sum0 += A[kk * M + (ii + 0)] * b_val;
        sum1 += A[kk * M + (ii + 1)] * b_val;
        sum2 += A[kk * M + (ii + 2)] * b_val;
        sum3 += A[kk * M + (ii + 3)] * b_val;
      }
      C[(ii + 0) * N + jj] += alpha * sum0;
      C[(ii + 1) * N + jj] += alpha * sum1;
      C[(ii + 2) * N + jj] += alpha * sum2;
      C[(ii + 3) * N + jj] += alpha * sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    size_t jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256d c_vec = _mm256_setzero_pd();
      for (size_t kk = k; kk < k_max; ++kk) {
        __m256d a_vec = _mm256_broadcast_sd(&A[kk * M + ii]);
        __m256d b_vec = _mm256_load_pd(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
      }
      __m256d c_old = _mm256_load_pd(&C[ii * N + jj]);
      _mm256_store_pd(&C[ii * N + jj], _mm256_fmadd_pd(alpha_vec, c_vec, c_old));
    }
    for (; jj < j_max; ++jj) {
      double sum = 0.0;
      for (size_t kk = k; kk < k_max; ++kk) {
        sum += A[kk * M + ii] * B[kk * N + jj];
      }
      C[ii * N + jj] += alpha * sum;
    }
  }
}
#endif

void dgemm(const double *A, const double *B, double *C, const size_t M, const size_t N,
           const size_t K, const bool trans_A, const bool trans_B, const double alpha,
           const double beta) {

  if (beta == 0.0) {
    parallel_for<size_t>(0, M * N, [&](size_t i) { C[i] = 0.0; });
  } else if (beta != 1.0) {
    parallel_for<size_t>(0, M * N, [&](size_t i) { C[i] *= beta; });
  }

#ifdef __AVX2__
  bool all_aligned = is_aligned_32(A) && is_aligned_32(B) && is_aligned_32(C);

  size_t M_BLOCK_SIZE, N_BLOCK_SIZE, K_BLOCK_SIZE;

  if (!trans_A && !trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 2;
    size_t M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    size_t N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

    if (all_aligned) {
      parallel_for_2d(M_blocks, N_blocks, [&](size_t block_i, size_t block_j) {
        size_t i_start = block_i * M_BLOCK_SIZE;
        size_t j_start = block_j * N_BLOCK_SIZE;
        size_t i_end = std::min(i_start + M_BLOCK_SIZE, M);
        size_t j_end = std::min(j_start + N_BLOCK_SIZE, N);
        for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
          size_t k_end = std::min(k + K_BLOCK_SIZE, K);
          dgemm_kernel_avx2_nn_aligned(A, B, C, M, N, K, i_start, j_start, k, i_end, j_end, k_end,
                                       alpha);
        }
      });
    } else {
      parallel_for_2d(M_blocks, N_blocks, [&](size_t block_i, size_t block_j) {
        size_t i_start = block_i * M_BLOCK_SIZE;
        size_t j_start = block_j * N_BLOCK_SIZE;
        size_t i_end = std::min(i_start + M_BLOCK_SIZE, M);
        size_t j_end = std::min(j_start + N_BLOCK_SIZE, N);
        for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
          size_t k_end = std::min(k + K_BLOCK_SIZE, K);
          dgemm_kernel_avx2_nn(A, B, C, M, N, K, i_start, j_start, k, i_end, j_end, k_end, alpha);
        }
      });
    }
  } else if (!trans_A && trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE / 2;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE / 2;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 16;

    size_t M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    size_t N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

    if (all_aligned) {
      parallel_for_2d(M_blocks, N_blocks, [&](size_t block_i, size_t block_j) {
        size_t i_start = block_i * M_BLOCK_SIZE;
        size_t j_start = block_j * N_BLOCK_SIZE;
        size_t i_end = std::min(i_start + M_BLOCK_SIZE, M);
        size_t j_end = std::min(j_start + N_BLOCK_SIZE, N);
        for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
          size_t k_end = std::min(k + K_BLOCK_SIZE, K);
          dgemm_kernel_avx2_nt_aligned(A, B, C, M, N, K, i_start, j_start, k, i_end, j_end, k_end,
                                       alpha);
        }
      });
    } else {
      parallel_for_2d(M_blocks, N_blocks, [&](size_t block_i, size_t block_j) {
        size_t i_start = block_i * M_BLOCK_SIZE;
        size_t j_start = block_j * N_BLOCK_SIZE;
        size_t i_end = std::min(i_start + M_BLOCK_SIZE, M);
        size_t j_end = std::min(j_start + N_BLOCK_SIZE, N);
        for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
          size_t k_end = std::min(k + K_BLOCK_SIZE, K);
          dgemm_kernel_avx2_nt(A, B, C, M, N, K, i_start, j_start, k, i_end, j_end, k_end, alpha);
        }
      });
    }
  } else if (trans_A && !trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 2;

    size_t M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    size_t N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

    if (all_aligned) {
      parallel_for_2d(M_blocks, N_blocks, [&](size_t block_i, size_t block_j) {
        size_t i_start = block_i * M_BLOCK_SIZE;
        size_t j_start = block_j * N_BLOCK_SIZE;
        size_t i_end = std::min(i_start + M_BLOCK_SIZE, M);
        size_t j_end = std::min(j_start + N_BLOCK_SIZE, N);
        for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
          size_t k_end = std::min(k + K_BLOCK_SIZE, K);
          dgemm_kernel_avx2_tn_aligned(A, B, C, M, N, K, i_start, j_start, k, i_end, j_end, k_end,
                                       alpha);
        }
      });
    } else {
      parallel_for_2d(M_blocks, N_blocks, [&](size_t block_i, size_t block_j) {
        size_t i_start = block_i * M_BLOCK_SIZE;
        size_t j_start = block_j * N_BLOCK_SIZE;
        size_t i_end = std::min(i_start + M_BLOCK_SIZE, M);
        size_t j_end = std::min(j_start + N_BLOCK_SIZE, N);
        for (size_t k = 0; k < K; k += K_BLOCK_SIZE) {
          size_t k_end = std::min(k + K_BLOCK_SIZE, K);
          dgemm_kernel_avx2_tn(A, B, C, M, N, K, i_start, j_start, k, i_end, j_end, k_end, alpha);
        }
      });
    }
  } else {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < K; ++k) {
          sum += A[k * M + i] * B[j * K + k];
        }
        C[i * N + j] += alpha * sum;
      }
    }
  }
#else
  const size_t BLOCK_SIZE = 32;

  if (beta == 0.0) {
    memset(C, 0, M * N * sizeof(double));
  } else if (beta != 1.0) {
    for (size_t i = 0; i < M * N; ++i) {
      C[i] *= beta;
    }
  }

  for (size_t i = 0; i < M; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N; j += BLOCK_SIZE) {
      for (size_t k = 0; k < K; k += BLOCK_SIZE) {
        size_t i_max = std::min(i + BLOCK_SIZE, M);
        size_t j_max = std::min(j + BLOCK_SIZE, N);
        size_t k_max = std::min(k + K_BLOCK_SIZE, K);

        for (size_t ii = i; ii < i_max; ++ii) {
          for (size_t jj = j; jj < j_max; ++jj) {
            double sum = 0.0;
            for (size_t kk = k; kk < k_max; ++kk) {
              size_t a_idx = trans_A ? kk * M + ii : ii * K + kk;
              size_t b_idx = trans_B ? jj * K + kk : kk * N + jj;
              sum += A[a_idx] * B[b_idx];
            }
            C[ii * N + jj] += alpha * sum;
          }
        }
      }
    }
  }
#endif
}

} // namespace cpu
} // namespace tnn