#include "math/gemm.hpp"
#include "tensor/tensor_extended.hpp"
#include "utils/misc.hpp"
#include "utils/parallel_for.hpp"
#ifdef USE_TBB
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#endif
constexpr size_t N = 64;
constexpr size_t C = 64;
constexpr size_t H = 32;
constexpr size_t W = 32;
constexpr size_t OC = 64;

using namespace tmath;
using namespace utils;

void benchmark(const Tensor<float, NCHW> &input, const size_t kernel_h, const size_t kernel_w,
               const size_t stride_h, const size_t stride_w, const size_t pad_h,
               const size_t pad_w) {
  std::cout << "Benchmarking im2col and col2im with kernel: " << kernel_h << "x" << kernel_w
            << ", stride: " << stride_h << "x" << stride_w << ", padding: " << pad_h << "x" << pad_w
            << std::endl;
  Matrix<float> col;
  benchmark("im2col",
            [&]() { col = im2col(input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w); });

  if (stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
#ifdef __AVX2__
    Matrix<float> col_opt;
    benchmark("optimized_im2col_stride_1_pad_0",
              [&]() { col_opt = optimized_im2col_stride_1_pad_0(input, kernel_h, kernel_w); });

    // verify
    if (col_opt.rows() != col.rows() || col_opt.cols() != col.cols()) {
      std::cerr << "Optimized im2col dimensions do not match!" << std::endl;
      return;
    }
    for (size_t i = 0; i < col.rows(); ++i) {
      for (size_t j = 0; j < col.cols(); ++j) {
        if (std::abs(col_opt(i, j) - col(i, j)) > 1e-5f) {
          std::cerr << "Mismatch at (" << i << ", " << j << "): " << col_opt(i, j) << " vs "
                    << col(i, j) << std::endl;
          return;
        }
      }
    }
    std::cout << "Optimized im2col matches standard im2col." << std::endl;
#endif
  }
  Tensor<float, NCHW> output;
  benchmark("col2im", [&]() {
    output = col2im(col, N, C, H, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  });
}

int main() {
#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(8));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&] {
#endif
    Tensor<float, NCHW> input(N, C, H, W);
    input.fill_random_uniform(1.0f);
    benchmark(input, 3, 3, 1, 1, 0, 0);
    benchmark(input, 3, 3, 1, 1, 1, 1);
    benchmark(input, 5, 5, 1, 1, 2, 2);
    benchmark(input, 3, 3, 2, 2, 1, 1);
    benchmark(input, 5, 5, 2, 2, 1, 1);
#ifdef USE_TBB
  });
#endif
  return 0;
}