#include "utils/ops.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace utils;

// Simple reference GEMM for verification
template <typename T>
void reference_gemm(const T *a, const T *b, T *c, size_t M, size_t N, size_t K) {
  parallel_for_2d(M, N, [&](size_t i, size_t j) {
    T sum = T(0);
    for (size_t k = 0; k < K; ++k) {
      sum += a[i * K + k] * b[k * N + j];
    }
    c[i * N + j] = sum;
  });
}

void current_gemm(const float *a, const float *b, float *c, size_t M, size_t N, size_t K) {
  float *b_transposed = (float *)malloc(sizeof(float) * K * N);
  transpose_2d(b, b_transposed, K, N);
  parallel_for_2d(M, N, [&](size_t i, size_t j) {
    c[i * N + j] = simd_dot_product(&a[i * K], &b_transposed[j * K], K);
  });
  free(b_transposed);
}

int main() {
  // Test dimensions
  const size_t M = 1024, N = 1024, K = 1024;

  std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);

  // Initialize with random values
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (size_t i = 0; i < M * K; ++i) {
    A[i] = dis(gen);
  }
  for (size_t i = 0; i < K * N; ++i) {
    B[i] = dis(gen);
  }

  // Initialize C matrices to zero
  std::fill(C.begin(), C.end(), 0.0f);
  std::fill(C_ref.begin(), C_ref.end(), 0.0f);

  auto reference_gemm_start = std::chrono::high_resolution_clock::now();
  // Compute reference result
  reference_gemm(A.data(), B.data(), C_ref.data(), M, N, K);
  auto reference_gemm_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> reference_duration = reference_gemm_end - reference_gemm_start;
  std::cout << "Reference GEMM time: " << reference_duration.count() << " seconds" << std::endl;

  auto current_gemm_start = std::chrono::high_resolution_clock::now();
  // Compute using current implementation
  current_gemm(A.data(), B.data(), C.data(), M, N, K);
  auto current_gemm_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> current_duration = current_gemm_end - current_gemm_start;
  std::cout << "Current GEMM time: " << current_duration.count() << " seconds" << std::endl;

  float max_current_error = 0.0f;
  for (size_t i = 0; i < M * N; ++i) {
    float error = std::abs(C[i] - C_ref[i]);
    max_current_error = std::max(max_current_error, error);
  }
  std::cout << "Current GEMM max error: " << max_current_error << std::endl;

  std::fill(C.begin(), C.end(), 0.0f);
  auto gemm_start = std::chrono::high_resolution_clock::now();
  // Compute using our optimized GEMM
  gemm(A.data(), B.data(), C.data(), M, N, K);
  auto gemm_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gemm_duration = gemm_end - gemm_start;
  std::cout << "Optimized GEMM time: " << gemm_duration.count() << " seconds" << std::endl;

  // Verify correctness
  float max_error = 0.0f;
  for (size_t i = 0; i < M * N; ++i) {
    float error = std::abs(C[i] - C_ref[i]);
    max_error = std::max(max_error, error);
  }

  std::cout << "GEMM Test Results:" << std::endl;
  std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x"
            << N << std::endl;
  std::cout << "Maximum error: " << max_error << std::endl;

  if (max_error < 1e-5f) {
    std::cout << "✅ GEMM implementation is correct!" << std::endl;
    return 0;
  } else {
    std::cout << "❌ GEMM implementation has errors!" << std::endl;
    return 1;
  }
}