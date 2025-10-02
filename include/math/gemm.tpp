#pragma once

#include "gemm.hpp"

namespace tmath {
constexpr int DEFAULT_BLOCK_SIZE = 32;

#ifdef __AVX2__
inline bool is_aligned_32(const void *ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0; }

inline void sgemm_kernel_avx2_nn(const float *A, const float *B, float *C, const int M, const int N,
                                 const int K, const int i, const int j, const int k,
                                 const int i_max, const int j_max, const int k_max) {
  int ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (int kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        __m256 a_vec_0 = _mm256_set1_ps(A[(ii + 0) * K + kk]);
        __m256 a_vec_1 = _mm256_set1_ps(A[(ii + 1) * K + kk]);
        __m256 a_vec_2 = _mm256_set1_ps(A[(ii + 2) * K + kk]);
        __m256 a_vec_3 = _mm256_set1_ps(A[(ii + 3) * K + kk]);
        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      _mm256_storeu_ps(&C[(ii + 0) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_storeu_ps(&C[(ii + 1) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_storeu_ps(&C[(ii + 2) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_storeu_ps(&C[(ii + 3) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[(ii + 0) * K + kk] * b_val;
        sum1 += A[(ii + 1) * K + kk] * b_val;
        sum2 += A[(ii + 2) * K + kk] * b_val;
        sum3 += A[(ii + 3) * K + kk] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (int kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_set1_ps(A[ii * K + kk]);
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_storeu_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj]), c_vec));
    }
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}

inline void sgemm_kernel_avx2_nn_aligned(const float *A, const float *B, float *C, const int M,
                                         const int N, const int K, const int i, const int j,
                                         const int k, const int i_max, const int j_max,
                                         const int k_max) {
  int ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (int kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        __m256 a_vec_0 = _mm256_set1_ps(A[(ii + 0) * K + kk]);
        __m256 a_vec_1 = _mm256_set1_ps(A[(ii + 1) * K + kk]);
        __m256 a_vec_2 = _mm256_set1_ps(A[(ii + 2) * K + kk]);
        __m256 a_vec_3 = _mm256_set1_ps(A[(ii + 3) * K + kk]);
        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      _mm256_store_ps(&C[(ii + 0) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_store_ps(&C[(ii + 1) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_store_ps(&C[(ii + 2) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_store_ps(&C[(ii + 3) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[(ii + 0) * K + kk] * b_val;
        sum1 += A[(ii + 1) * K + kk] * b_val;
        sum2 += A[(ii + 2) * K + kk] * b_val;
        sum3 += A[(ii + 3) * K + kk] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (int kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_set1_ps(A[ii * K + kk]);
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_store_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_load_ps(&C[ii * N + jj]), c_vec));
    }
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}

// Transpose B (NT): C = A * B^T
inline void sgemm_kernel_avx2_nt(const float *A, const float *B, float *C, const int M, const int N,
                                 const int K, const int i, const int j, const int k,
                                 const int i_max, const int j_max, const int k_max) {
  for (int ii = i; ii < i_max; ++ii) {
    int jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();

      int kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[ii * K + kk]);
        __m256 b0_vec = _mm256_loadu_ps(&B[(jj + 0) * K + kk]);
        __m256 b1_vec = _mm256_loadu_ps(&B[(jj + 1) * K + kk]);
        __m256 b2_vec = _mm256_loadu_ps(&B[(jj + 2) * K + kk]);
        __m256 b3_vec = _mm256_loadu_ps(&B[(jj + 3) * K + kk]);

        sum0 = _mm256_fmadd_ps(a_vec, b0_vec, sum0);
        sum1 = _mm256_fmadd_ps(a_vec, b1_vec, sum1);
        sum2 = _mm256_fmadd_ps(a_vec, b2_vec, sum2);
        sum3 = _mm256_fmadd_ps(a_vec, b3_vec, sum3);
      }

      // Horizontal sum for each accumulator
      auto horizontal_sum = [](const __m256 &vec) -> float {
        __m128 vlow = _mm256_castps256_ps128(vec);
        __m128 vhigh = _mm256_extractf128_ps(vec, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        return _mm_cvtss_f32(vlow);
      };

      float partial_sum0 = horizontal_sum(sum0);
      float partial_sum1 = horizontal_sum(sum1);
      float partial_sum2 = horizontal_sum(sum2);
      float partial_sum3 = horizontal_sum(sum3);

      for (; kk < k_max; ++kk) {
        float a_val = A[ii * K + kk];
        partial_sum0 += a_val * B[(jj + 0) * K + kk];
        partial_sum1 += a_val * B[(jj + 1) * K + kk];
        partial_sum2 += a_val * B[(jj + 2) * K + kk];
        partial_sum3 += a_val * B[(jj + 3) * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj + 0] = partial_sum0;
        C[ii * N + jj + 1] = partial_sum1;
        C[ii * N + jj + 2] = partial_sum2;
        C[ii * N + jj + 3] = partial_sum3;
      } else {
        C[ii * N + jj + 0] += partial_sum0;
        C[ii * N + jj + 1] += partial_sum1;
        C[ii * N + jj + 2] += partial_sum2;
        C[ii * N + jj + 3] += partial_sum3;
      }
    }

    for (; jj < j_max; ++jj) {
      __m256 sum_vec = _mm256_setzero_ps();
      int kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[ii * K + kk]);
        __m256 b_vec = _mm256_loadu_ps(&B[jj * K + kk]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }

      __m128 vlow = _mm256_castps256_ps128(sum_vec);
      __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
      vlow = _mm_add_ps(vlow, vhigh);
      vlow = _mm_hadd_ps(vlow, vlow);
      vlow = _mm_hadd_ps(vlow, vlow);
      float sum = _mm_cvtss_f32(vlow);

      for (; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[jj * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj] = sum;
      } else {
        C[ii * N + jj] += sum;
      }
    }
  }
}

// Aligned version: Transpose B (NT): C = A * B^T
inline void sgemm_kernel_avx2_nt_aligned(const float *A, const float *B, float *C, const int M,
                                         const int N, const int K, const int i, const int j,
                                         const int k, const int i_max, const int j_max,
                                         const int k_max) {
  for (int ii = i; ii < i_max; ++ii) {
    int jj = j;
    for (; jj + 3 < j_max; jj += 4) {
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();

      int kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_load_ps(&A[ii * K + kk]);
        __m256 b0_vec = _mm256_load_ps(&B[(jj + 0) * K + kk]);
        __m256 b1_vec = _mm256_load_ps(&B[(jj + 1) * K + kk]);
        __m256 b2_vec = _mm256_load_ps(&B[(jj + 2) * K + kk]);
        __m256 b3_vec = _mm256_load_ps(&B[(jj + 3) * K + kk]);

        sum0 = _mm256_fmadd_ps(a_vec, b0_vec, sum0);
        sum1 = _mm256_fmadd_ps(a_vec, b1_vec, sum1);
        sum2 = _mm256_fmadd_ps(a_vec, b2_vec, sum2);
        sum3 = _mm256_fmadd_ps(a_vec, b3_vec, sum3);
      }

      // Horizontal sum for each accumulator
      auto horizontal_sum = [](const __m256 &vec) -> float {
        __m128 vlow = _mm256_castps256_ps128(vec);
        __m128 vhigh = _mm256_extractf128_ps(vec, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        return _mm_cvtss_f32(vlow);
      };

      float partial_sum0 = horizontal_sum(sum0);
      float partial_sum1 = horizontal_sum(sum1);
      float partial_sum2 = horizontal_sum(sum2);
      float partial_sum3 = horizontal_sum(sum3);

      for (; kk < k_max; ++kk) {
        float a_val = A[ii * K + kk];
        partial_sum0 += a_val * B[(jj + 0) * K + kk];
        partial_sum1 += a_val * B[(jj + 1) * K + kk];
        partial_sum2 += a_val * B[(jj + 2) * K + kk];
        partial_sum3 += a_val * B[(jj + 3) * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj + 0] = partial_sum0;
        C[ii * N + jj + 1] = partial_sum1;
        C[ii * N + jj + 2] = partial_sum2;
        C[ii * N + jj + 3] = partial_sum3;
      } else {
        C[ii * N + jj + 0] += partial_sum0;
        C[ii * N + jj + 1] += partial_sum1;
        C[ii * N + jj + 2] += partial_sum2;
        C[ii * N + jj + 3] += partial_sum3;
      }
    }

    for (; jj < j_max; ++jj) {
      __m256 sum_vec = _mm256_setzero_ps();
      int kk = k;
      for (; kk + 7 < k_max; kk += 8) {
        __m256 a_vec = _mm256_load_ps(&A[ii * K + kk]);
        __m256 b_vec = _mm256_load_ps(&B[jj * K + kk]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }

      __m128 vlow = _mm256_castps256_ps128(sum_vec);
      __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
      vlow = _mm_add_ps(vlow, vhigh);
      vlow = _mm_hadd_ps(vlow, vlow);
      vlow = _mm_hadd_ps(vlow, vlow);
      float sum = _mm_cvtss_f32(vlow);

      for (; kk < k_max; ++kk) {
        sum += A[ii * K + kk] * B[jj * K + kk];
      }

      if (k == 0) {
        C[ii * N + jj] = sum;
      } else {
        C[ii * N + jj] += sum;
      }
    }
  }
}

// Transpose A (TN): C = A^T * B (Optimized with 4x8 micro-kernel)
inline void sgemm_kernel_avx2_tn(const float *A, const float *B, float *C, const int M, const int N,
                                 const int K, const int i, const int j, const int k,
                                 const int i_max, const int j_max, const int k_max) {
  int ii = i;
  // Process 4 rows of C at a time
  for (; ii + 3 < i_max; ii += 4) {
    int jj = j;
    // Process 8 columns of C at a time
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (int kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);

        __m256 a_vec_0 = _mm256_set1_ps(A[kk * M + ii + 0]);
        __m256 a_vec_1 = _mm256_set1_ps(A[kk * M + ii + 1]);
        __m256 a_vec_2 = _mm256_set1_ps(A[kk * M + ii + 2]);
        __m256 a_vec_3 = _mm256_set1_ps(A[kk * M + ii + 3]);

        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      // Add accumulated values to C
      _mm256_storeu_ps(&C[(ii + 0) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_storeu_ps(&C[(ii + 1) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_storeu_ps(&C[(ii + 2) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_storeu_ps(&C[(ii + 3) * N + jj],
                       _mm256_add_ps(_mm256_loadu_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[kk * M + ii + 0] * b_val;
        sum1 += A[kk * M + ii + 1] * b_val;
        sum2 += A[kk * M + ii + 2] * b_val;
        sum3 += A[kk * M + ii + 3] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (int kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_set1_ps(A[kk * M + ii]);
        __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_storeu_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj]), c_vec));
    }
    // Scalar remainder for the final rows
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        sum += A[kk * M + ii] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}

// Aligned version: Transpose A (TN): C = A^T * B (Optimized with 4x8 micro-kernel)
inline void sgemm_kernel_avx2_tn_aligned(const float *A, const float *B, float *C, const int M,
                                         const int N, const int K, const int i, const int j,
                                         const int k, const int i_max, const int j_max,
                                         const int k_max) {
  int ii = i;
  for (; ii + 3 < i_max; ii += 4) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec_0 = _mm256_setzero_ps();
      __m256 c_vec_1 = _mm256_setzero_ps();
      __m256 c_vec_2 = _mm256_setzero_ps();
      __m256 c_vec_3 = _mm256_setzero_ps();

      for (int kk = k; kk < k_max; ++kk) {
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        __m256 a_vec_0 = _mm256_set1_ps(A[kk * M + ii + 0]);
        __m256 a_vec_1 = _mm256_set1_ps(A[kk * M + ii + 1]);
        __m256 a_vec_2 = _mm256_set1_ps(A[kk * M + ii + 2]);
        __m256 a_vec_3 = _mm256_set1_ps(A[kk * M + ii + 3]);
        c_vec_0 = _mm256_fmadd_ps(a_vec_0, b_vec, c_vec_0);
        c_vec_1 = _mm256_fmadd_ps(a_vec_1, b_vec, c_vec_1);
        c_vec_2 = _mm256_fmadd_ps(a_vec_2, b_vec, c_vec_2);
        c_vec_3 = _mm256_fmadd_ps(a_vec_3, b_vec, c_vec_3);
      }
      _mm256_store_ps(&C[(ii + 0) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 0) * N + jj]), c_vec_0));
      _mm256_store_ps(&C[(ii + 1) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 1) * N + jj]), c_vec_1));
      _mm256_store_ps(&C[(ii + 2) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 2) * N + jj]), c_vec_2));
      _mm256_store_ps(&C[(ii + 3) * N + jj],
                      _mm256_add_ps(_mm256_load_ps(&C[(ii + 3) * N + jj]), c_vec_3));
    }
    for (; jj < j_max; ++jj) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        float b_val = B[kk * N + jj];
        sum0 += A[kk * M + ii + 0] * b_val;
        sum1 += A[kk * M + ii + 1] * b_val;
        sum2 += A[kk * M + ii + 2] * b_val;
        sum3 += A[kk * M + ii + 3] * b_val;
      }
      C[(ii + 0) * N + jj] += sum0;
      C[(ii + 1) * N + jj] += sum1;
      C[(ii + 2) * N + jj] += sum2;
      C[(ii + 3) * N + jj] += sum3;
    }
  }
  for (; ii < i_max; ++ii) {
    int jj = j;
    for (; jj + 7 < j_max; jj += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (int kk = k; kk < k_max; ++kk) {
        __m256 a_vec = _mm256_set1_ps(A[kk * M + ii]);
        __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      }
      _mm256_store_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_load_ps(&C[ii * N + jj]), c_vec));
    }
    for (; jj < j_max; ++jj) {
      float sum = 0.0f;
      for (int kk = k; kk < k_max; ++kk) {
        sum += A[kk * M + ii] * B[kk * N + jj];
      }
      C[ii * N + jj] += sum;
    }
  }
}
#endif

void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K,
           const bool trans_A, const bool trans_B) {
#ifdef __AVX2__
  // check for AVX 32 bit alignment
  bool all_aligned = is_aligned_32(A) && is_aligned_32(B) && is_aligned_32(C);

  int M_BLOCK_SIZE, N_BLOCK_SIZE, K_BLOCK_SIZE;

  if (!trans_A && !trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 2;
    int M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    int N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
    // NN: C = A * B
    if (all_aligned) {
      utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
        int i = block_i * M_BLOCK_SIZE;
        int j = block_j * N_BLOCK_SIZE;
        for (int k = 0; k < K; k += K_BLOCK_SIZE) {
          int i_max = std::min(i + M_BLOCK_SIZE, M);
          int j_max = std::min(j + N_BLOCK_SIZE, N);
          int k_max = std::min(k + K_BLOCK_SIZE, K);
          sgemm_kernel_avx2_nn_aligned(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
        }
      });
    } else {
      utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
        int i = block_i * M_BLOCK_SIZE;
        int j = block_j * N_BLOCK_SIZE;
        for (int k = 0; k < K; k += K_BLOCK_SIZE) {
          int i_max = std::min(i + M_BLOCK_SIZE, M);
          int j_max = std::min(j + N_BLOCK_SIZE, N);
          int k_max = std::min(k + K_BLOCK_SIZE, K);
          sgemm_kernel_avx2_nn(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
        }
      });
    }
  } else if (!trans_A && trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE / 2;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE / 2;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 8;

    int M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    int N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
    // NT: C = A * B^T
    if (all_aligned) {
      utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
        int i = block_i * M_BLOCK_SIZE;
        int j = block_j * N_BLOCK_SIZE;
        for (int k = 0; k < K; k += K_BLOCK_SIZE) {
          int i_max = std::min(i + M_BLOCK_SIZE, M);
          int j_max = std::min(j + N_BLOCK_SIZE, N);
          int k_max = std::min(k + K_BLOCK_SIZE, K);
          sgemm_kernel_avx2_nt_aligned(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
        }
      });
    } else {
      utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
        int i = block_i * M_BLOCK_SIZE;
        int j = block_j * N_BLOCK_SIZE;
        for (int k = 0; k < K; k += K_BLOCK_SIZE) {
          int i_max = std::min(i + M_BLOCK_SIZE, M);
          int j_max = std::min(j + N_BLOCK_SIZE, N);
          int k_max = std::min(k + K_BLOCK_SIZE, K);
          sgemm_kernel_avx2_nt(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
        }
      });
    }
  } else if (trans_A && !trans_B) {
    M_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    N_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    K_BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
    int M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    int N_blocks = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
    // TN: C = A^T * B
    if (all_aligned) {
      utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
        int i = block_i * M_BLOCK_SIZE;
        int j = block_j * N_BLOCK_SIZE;
        for (int k = 0; k < K; k += K_BLOCK_SIZE) {
          int i_max = std::min(i + M_BLOCK_SIZE, M);
          int j_max = std::min(j + N_BLOCK_SIZE, N);
          int k_max = std::min(k + K_BLOCK_SIZE, K);
          sgemm_kernel_avx2_tn_aligned(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
        }
      });
    } else {
      utils::parallel_for_2d(M_blocks, N_blocks, [&](int block_i, int block_j) {
        int i = block_i * M_BLOCK_SIZE;
        int j = block_j * N_BLOCK_SIZE;
        for (int k = 0; k < K; k += K_BLOCK_SIZE) {
          int i_max = std::min(i + M_BLOCK_SIZE, M);
          int j_max = std::min(j + N_BLOCK_SIZE, N);
          int k_max = std::min(k + K_BLOCK_SIZE, K);
          sgemm_kernel_avx2_tn(A, B, C, M, N, K, i, j, k, i_max, j_max, k_max);
        }
      });
    }
  } else {
    // TT: C = A^T * B^T <-> (A*B)^T = C
    float *D = (float *)aligned_alloc(32, sizeof(float) * N * M);
    sgemm(B, A, D, N, M, K, false, false);
    utils::transpose_2d(D, C, N, M);
    free(D);
  }
#else
  // Scalar fallback implementation
  for (int i = 0; i < M; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
      for (int k = 0; k < K; k += BLOCK_SIZE) {
        int i_max = std::min(i + BLOCK_SIZE, M);
        int j_max = std::min(j + BLOCK_SIZE, N);
        int k_max = std::min(k + BLOCK_SIZE, K);
        for (int ii = i; ii < i_max; ++ii) {
          for (int jj = j; jj < j_max; ++jj) {
            float sum = C[ii * N + jj];
            for (int kk = k; kk < k_max; ++kk) {
              float a_val = trans_A ? A[kk * M + ii] : A[ii * K + kk];
              float b_val = trans_B ? B[jj * K + kk] : B[kk * N + jj];
              sum += a_val * b_val;
            }
            C[ii * N + jj] = sum;
          }
        }
      }
    }
  }
#endif
}

void old_gemm(const float *A, const float *B, float *C, const int M, const int N, const int K) {
  float *B_T = (float *)malloc(sizeof(float) * K * N);
  utils::transpose_2d(B, B_T, K, N);
  utils::parallel_for_2d(M, N, [&](int i, int j) {
    float sum = utils::simd_dot_product(&A[i * K], &B_T[j * K], K);
    C[i * N + j] = sum;
  });
  free(B_T);
}
} // namespace tmath