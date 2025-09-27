#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "tensor/tensor.hpp"
#include "utils/ops.hpp"

using namespace utils;

void gemm_1() {
  Tensor<float> A(64, 1, 28, 28);
  Matrix<float> W(8, 1 * 5 * 5);
  A.fill_random_normal(0.5f, 0.25f);
  W.fill_random_normal(0.0f, 1.0f);
  Matrix<float> A_col = A.im2col(5, 5, 1, 1, 0, 0);
  Matrix<float> output(8, 64 * 24 * 24);
  std::cout << "A_col dims:" << A_col.rows() << " x " << A_col.cols() << std::endl;
  std::cout << "W dims:" << W.rows() << " x " << W.cols() << std::endl;
  output.fill(0.0f);
  auto start = std::chrono::high_resolution_clock::now();
  matmul(A_col.data(), W.data(), output.data(), 64 * 24 * 24, 8, 1 * 5 * 5);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "GEMM time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds" << std::endl;
}

void gemm_2() {
  Tensor<float> B(64, 16, 8, 8);
  Matrix<float> W(48, 16 * 5 * 5);
  B.fill_random_normal(0.5f, 0.25f);
  W.fill_random_normal(0.0f, 1.0f);
  Matrix<float> B_col = B.im2col(5, 5, 1, 1, 0, 0);
  Matrix<float> output(48, 64 * 4 * 4);
  std::cout << "B_col dims:" << B_col.rows() << " x " << B_col.cols() << std::endl;
  std::cout << "W dims:" << W.rows() << " x " << W.cols() << std::endl;
  output.fill(0.0f);
  auto start = std::chrono::high_resolution_clock::now();
  matmul(B_col.data(), W.data(), output.data(), 64 * 4 * 4, 48, 16 * 5 * 5);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "GEMM time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds" << std::endl;
}

int main() {
  gemm_1();
  gemm_2();
  return 0;
}