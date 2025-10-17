#include "math/gemm.hpp"
#include "tensor/tensor_extended.hpp"
#include "utils/misc.hpp"
constexpr size_t N = 64;
constexpr size_t C = 128;
constexpr size_t H = 32;
constexpr size_t W = 32;

using namespace tmath;
using namespace utils;

void benchmark(const Tensor<float, NCHW> &input, const size_t kernel_h, const size_t kernel_w,
               const size_t stride_h, const size_t stride_w, const size_t pad_h,
               const size_t pad_w) {
  std::cout << "Benchmarking im2col and col2im with kernel: " << kernel_h << "x" << kernel_w
            << ", stride: " << stride_h << "x" << stride_w << ", padding: " << pad_h << "x" << pad_w
            << std::endl;
  Matrix<float> col;
  benchmark("im2col padded", [&]() {
    col = im2col_padded(input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  });

  Matrix<float> col2;
  benchmark("im2col 3x3 pad 1 stride 1", [&]() { col2 = im2col_pad_1_stride_1_kernel_3(input); });

  Matrix<float> col3;
  benchmark("im2col explicit padding", [&]() {
    col3 = im2col(input.pad(pad_h, pad_w), kernel_h, kernel_w, stride_h, stride_w, 0, 0);
  });

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
    // benchmark(input, 3, 3, 1, 1, 0, 0);
    benchmark(input, 3, 3, 1, 1, 1, 1);
    // benchmark(input, 5, 5, 1, 1, 2, 2);
    // benchmark(input, 3, 3, 2, 2, 1, 1);
    // benchmark(input, 5, 5, 2, 2, 1, 1);
#ifdef USE_TBB
  });
#endif
  return 0;
}