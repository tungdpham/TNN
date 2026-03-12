#include "nn/layers_impl/cpu/slice_ops.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include "nn/layers_impl/cuda/slice_ops.hpp"
#endif

#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <vector>

namespace {

static size_t product(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

template <typename T>
static void reference_slice_forward(const std::vector<T> &input, std::vector<T> &output,
                                    const std::vector<size_t> &input_shape, size_t axis,
                                    size_t start, size_t length) {
  size_t outer_size = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size *= input_shape[i];
  }

  size_t axis_size = input_shape[axis];
  output.assign(outer_size * length * inner_size, T(0));

  for (size_t o = 0; o < outer_size; ++o) {
    for (size_t l = 0; l < length; ++l) {
      for (size_t i = 0; i < inner_size; ++i) {
        size_t output_idx = o * length * inner_size + l * inner_size + i;
        size_t input_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
        output[output_idx] = input[input_idx];
      }
    }
  }
}

template <typename T>
static void reference_slice_backward(const std::vector<T> &grad_output, std::vector<T> &grad_input,
                                     const std::vector<size_t> &input_shape, size_t axis,
                                     size_t start, size_t length) {
  size_t outer_size = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size *= input_shape[i];
  }

  size_t axis_size = input_shape[axis];
  grad_input.assign(product(input_shape), T(0));

  for (size_t o = 0; o < outer_size; ++o) {
    for (size_t l = 0; l < length; ++l) {
      for (size_t i = 0; i < inner_size; ++i) {
        size_t grad_idx = o * length * inner_size + l * inner_size + i;
        size_t input_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
        grad_input[input_idx] = grad_output[grad_idx];
      }
    }
  }
}

}  // namespace

TEST(SliceOpsTest, CpuForwardBackwardMatchesReference) {
  using T = float;

  const std::vector<size_t> input_shape = {2, 3, 4};
  std::vector<T> input(product(input_shape));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<T>(i + 1);
  }

  struct Case {
    size_t axis;
    size_t start;
    size_t length;
  };

  const std::vector<Case> cases = {
      {0, 0, 1},
      {0, 1, 1},
      {1, 1, 2},
      {2, 1, 2},
  };

  for (const auto &c : cases) {
    std::vector<T> expected_out;
    reference_slice_forward(input, expected_out, input_shape, c.axis, c.start, c.length);

    std::vector<T> actual_out(expected_out.size(), T(0));
    tnn::cpu::slice::slice_forward<T>(input.data(), actual_out.data(), input_shape, c.axis, c.start,
                                      c.length);

    ASSERT_EQ(actual_out.size(), expected_out.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_EQ(actual_out[i], expected_out[i])
          << "axis=" << c.axis << " start=" << c.start << " length=" << c.length << " idx=" << i;
    }

    std::vector<T> grad_output(expected_out.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
      grad_output[i] = static_cast<T>(100 + i);
    }

    std::vector<T> expected_grad_input;
    reference_slice_backward(grad_output, expected_grad_input, input_shape, c.axis, c.start,
                             c.length);

    std::vector<T> actual_grad_input(product(input_shape), T(-1));
    tnn::cpu::slice::slice_backward<T>(grad_output.data(), actual_grad_input.data(), input_shape,
                                       c.axis, c.start, c.length);

    ASSERT_EQ(actual_grad_input.size(), expected_grad_input.size());
    for (size_t i = 0; i < expected_grad_input.size(); ++i) {
      EXPECT_EQ(actual_grad_input[i], expected_grad_input[i])
          << "axis=" << c.axis << " start=" << c.start << " length=" << c.length << " idx=" << i;
    }
  }
}

#ifdef USE_CUDA
TEST(SliceOpsTest, CudaForwardBackwardMatchesReference) {
  using T = float;

  const std::vector<size_t> input_shape = {2, 3, 4};
  std::vector<T> input(product(input_shape));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<T>(i + 1);
  }

  const size_t axis = 1;
  const size_t start = 1;
  const size_t length = 2;

  std::vector<T> expected_out;
  reference_slice_forward(input, expected_out, input_shape, axis, start, length);

  T *d_in = nullptr;
  T *d_out = nullptr;
  cudaMalloc(&d_in, input.size() * sizeof(T));
  cudaMalloc(&d_out, expected_out.size() * sizeof(T));

  cudaMemcpy(d_in, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, expected_out.size() * sizeof(T));

  tnn::cuda::slice::slice_forward<T>(d_in, d_out, input_shape, axis, start, length, 0);

  std::vector<T> actual_out(expected_out.size(), T(0));
  cudaMemcpy(actual_out.data(), d_out, expected_out.size() * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_EQ(actual_out[i], expected_out[i]) << "idx=" << i;
  }

  std::vector<T> grad_output(expected_out.size());
  for (size_t i = 0; i < grad_output.size(); ++i) {
    grad_output[i] = static_cast<T>(100 + i);
  }
  std::vector<T> expected_grad_input;
  reference_slice_backward(grad_output, expected_grad_input, input_shape, axis, start, length);

  T *d_grad = nullptr;
  T *d_grad_in = nullptr;
  cudaMalloc(&d_grad, grad_output.size() * sizeof(T));
  cudaMalloc(&d_grad_in, input.size() * sizeof(T));

  cudaMemcpy(d_grad, grad_output.data(), grad_output.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_grad_in, 0, input.size() * sizeof(T));

  tnn::cuda::slice::slice_backward<T>(d_grad, d_grad_in, input_shape, axis, start, length, 0);

  std::vector<T> actual_grad_input(input.size(), T(0));
  cudaMemcpy(actual_grad_input.data(), d_grad_in, input.size() * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < expected_grad_input.size(); ++i) {
    EXPECT_EQ(actual_grad_input[i], expected_grad_input[i]) << "idx=" << i;
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_grad);
  cudaFree(d_grad_in);
}
#endif
