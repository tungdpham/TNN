#include "math/gemm.hpp"
#include "tensor/tensor.hpp"
#include "utils/misc.hpp"
#include "utils/ops.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <utils/mkl_utils.hpp>
#include <vector>

using namespace tmath;
using namespace utils;

constexpr size_t N = 64;
constexpr size_t C = 128;
constexpr size_t H = 128;
constexpr size_t W = 128;

bool check_match(const float *a, const float *b, size_t size, float max_acceptable_error = 1.0f) {
  float max_error = 0;
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(a[i] - b[i]) > 1e-3f) {
      max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
  }

  std::cout << "Max error: " << max_error << std::endl;
  if (max_error > max_acceptable_error) {
    return false;
  }
  return true;
}

int main() {

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(8));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&] {
#endif
    size_t n = N * C;
    size_t m = N * C;
    size_t k = H * W;

    std::cout << "Matrix A: " << n << " x " << k << std::endl;
    std::cout << "Matrix B: " << k << " x " << m << std::endl;
    std::cout << "Matrix C: " << n << " x " << m << std::endl;
    std::cout << "FLOP count: " << 2.0 * n * m * k << std::endl;

    Matrix<float> a(n, k);
    Matrix<float> b(k, m);
    Matrix<float> c1(n, m);
    Matrix<float> c2(n, m);
    Matrix<float> c3(n, m);
    Matrix<float> c4(n, m);
    Matrix<float> c1_optimized(n, m);
    Matrix<float> c2_optimized(n, m);
    Matrix<float> c3_optimized(n, m);
    Matrix<float> c4_optimized(n, m);
    Matrix<float> c1_mkl(n, m);
    Matrix<float> c2_mkl(n, m);
    Matrix<float> c3_mkl(n, m);
    Matrix<float> c4_mkl(n, m);

    a.fill_random_normal(0.5f, 0.25f);
    b.fill_random_normal(0.0f, 1.0f);
    c1.fill(0.0f);
    c2.fill(0.0f);
    c3.fill(0.0f);
    c4.fill(0.0f);

    benchmark(
        "SGEMM (NN)",
        [&]() { sgemm(a.data(), b.data(), c1.data(), N, C, C * H * W, false, false); }, 3);

    benchmark(
        "SGEMM (NT)", [&]() { sgemm(a.data(), b.data(), c2.data(), N, C, C * H * W, false, true); },
        3);

    benchmark(
        "SGEMM (TN)", [&]() { sgemm(a.data(), b.data(), c3.data(), N, C, C * H * W, true, false); },
        3);

    benchmark(
        "SGEMM (TT)", [&]() { sgemm(a.data(), b.data(), c4.data(), N, C, C * H * W, true, true); },
        3);
#ifdef USE_MKL
    std::cout << "\n=== MKL Benchmarks ===" << std::endl;
    mkl_set_num_threads(8);

    benchmark(
        "MKL SGEMM (NN)",
        [&]() {
          utils::mkl::gemm('N', 'N', N, C, C * H * W, 1.0f, a.data(), C * H * W, b.data(), C, 1.0f,
                           c1_mkl.data(), C);
        },
        3);

    benchmark(
        "MKL SGEMM (NT)",
        [&]() {
          utils::mkl::gemm('N', 'T', N, C, C * H * W, 1.0f, a.data(), C * H * W, b.data(),
                           C * H * W, 1.0f, c2_mkl.data(), C);
        },
        3);

    benchmark(
        "MKL SGEMM (TN)",
        [&]() {
          utils::mkl::gemm('T', 'N', N, C, C * H * W, 1.0f, a.data(), N, b.data(), C, 1.0f,
                           c3_mkl.data(), C);
        },
        3);

    benchmark(
        "MKL SGEMM (TT)",
        [&]() {
          utils::mkl::gemm('T', 'T', N, C, C * H * W, 1.0f, a.data(), N, b.data(), C * H * W, 1.0f,
                           c4_mkl.data(), C);
        },
        3);

    if (!check_match(c1.data(), c1_mkl.data(), N * C)) {
      std::cout << "Mismatch in C1 (NN)!" << std::endl;
    } else {
      std::cout << "C1 (NN) matches MKL result." << std::endl;
    }

    if (!check_match(c2.data(), c2_mkl.data(), N * C)) {
      std::cout << "Mismatch in C2 (NT)!" << std::endl;
    } else {
      std::cout << "C2 (NT) matches MKL result." << std::endl;
    }

    if (!check_match(c3.data(), c3_mkl.data(), N * C)) {
      std::cout << "Mismatch in C3 (TN)!" << std::endl;
    } else {
      std::cout << "C3 (TN) matches MKL result." << std::endl;
    }

    if (!check_match(c4.data(), c4_mkl.data(), N * C)) {
      std::cout << "Mismatch in C4 (TT)!" << std::endl;
    } else {
      std::cout << "C4 (TT) matches MKL result." << std::endl;
    }
#endif

#ifdef USE_TBB
  });
#endif

  return 0;
}