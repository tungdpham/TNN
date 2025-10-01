#include "math/gemm.hpp"
#include "tensor/tensor.hpp"
#include "utils/ops.hpp"
#include <chrono>
#include <iostream>
#include <utils/mkl_utils.hpp>

using namespace tmath;
using namespace utils;

constexpr int N = 64;
constexpr int C = 128;
constexpr int H = 128;
constexpr int W = 128;

bool check_match(const float *a, const float *b, size_t size, float tol = 0.1f) {
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(a[i] - b[i]) > tol) {
      std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

int main() {

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(8));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&] {
#endif
    Matrix<float> a(N, C * H * W);
    Matrix<float> b(C * H * W, C);
    Matrix<float> c1(N, C);
    Matrix<float> c2(N, C);
    Matrix<float> c3(N, C);
    Matrix<float> c4(N, C);
    Matrix<float> c1_mkl(N, C);
    Matrix<float> c2_mkl(N, C);
    Matrix<float> c3_mkl(N, C);
    Matrix<float> c4_mkl(N, C);

    a.fill_random_normal(0.5f, 0.25f);
    b.fill_random_normal(0.0f, 1.0f);
    c1.fill(0.0f);
    c2.fill(0.0f);
    c3.fill(0.0f);
    c4.fill(0.0f);

    auto current_start = std::chrono::high_resolution_clock::now();
    sgemm(a.data(), b.data(), c1.data(), N, C, C * H * W, false, false);
    auto current_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> current_duration = current_end - current_start;
    std::cout << "SGEMM completed in " << current_duration.count() << " ms\n";

    auto current_nt_start = std::chrono::high_resolution_clock::now();
    sgemm(a.data(), b.data(), c2.data(), N, C, C * H * W, false, true);
    auto current_nt_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> current_nt_duration =
        current_nt_end - current_nt_start;
    std::cout << "SGEMM (B^T) completed in " << current_nt_duration.count() << " ms\n";

    auto current_tn_start = std::chrono::high_resolution_clock::now();
    sgemm(a.data(), b.data(), c3.data(), N, C, C * H * W, true, false);
    auto current_tn_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> current_tn_duration =
        current_tn_end - current_tn_start;
    std::cout << "SGEMM (A^T) completed in " << current_tn_duration.count() << " ms\n";

    auto current_tt_start = std::chrono::high_resolution_clock::now();
    sgemm(a.data(), b.data(), c4.data(), N, C, C * H * W, true, true);
    auto current_tt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_tt_duration =
        current_tt_end - current_tt_start;
    std::cout << "SGEMM (A^T, B^T) completed in " << current_tt_duration.count() << " ms\n";

#ifdef USE_MKL
    mkl_set_threading_layer(MKL_THREADING_TBB);
    auto mkl_start = std::chrono::high_resolution_clock::now();
    utils::mkl::gemm('N', 'N', N, C, C * H * W, 1.0f, a.data(), C * H * W, b.data(), C, 0.0f,
                     c1_mkl.data(), C);
    auto mkl_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mkl_duration = mkl_end - mkl_start;
    std::cout << "MKL SGEMM completed in " << mkl_duration.count() << " ms\n";

    auto mkl_nt_start = std::chrono::high_resolution_clock::now();
    utils::mkl::gemm('N', 'T', N, C, C * H * W, 1.0f, a.data(), C * H * W, b.data(), C * H * W,
                     0.0f, c2_mkl.data(), C);
    auto mkl_nt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mkl_nt_duration = mkl_nt_end - mkl_nt_start;
    std::cout << "MKL SGEMM (B^T) completed in " << mkl_nt_duration.count() << " ms\n";

    auto mkl_tn_start = std::chrono::high_resolution_clock::now();
    utils::mkl::gemm('T', 'N', N, C, C * H * W, 1.0f, a.data(), N, b.data(), C, 0.0f, c3_mkl.data(),
                     C);
    auto mkl_tn_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mkl_tn_duration = mkl_tn_end - mkl_tn_start;
    std::cout << "MKL SGEMM (A^T) completed in " << mkl_tn_duration.count() << " ms\n";

    auto mkl_tt_start = std::chrono::high_resolution_clock::now();
    utils::mkl::gemm('T', 'T', N, C, C * H * W, 1.0f, a.data(), N, b.data(), C * H * W, 0.0f,
                     c4_mkl.data(), C);
    auto mkl_tt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mkl_tt_duration = mkl_tt_end - mkl_tt_start;
    std::cout << "MKL SGEMM (A^T, B^T) completed in " << mkl_tt_duration.count() << " ms\n";

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