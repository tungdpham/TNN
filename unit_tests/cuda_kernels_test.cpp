#ifdef USE_CUDA

#include "ops/cuda/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

class CudaKernelsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Setup test data
    size = 1000;
    host_a.resize(size);
    host_b.resize(size);
    host_c.resize(size);
    host_expected.resize(size);

    // Initialize with test values
    for (size_t i = 0; i < size; ++i) {
      host_a[i] = static_cast<float>(i * 0.1f);
      host_b[i] = static_cast<float>((i + 1) * 0.2f);
      host_c[i] = 0.0f;
    }

    // Allocate device memory
    cudaMalloc(&dev_a, size * sizeof(float));
    cudaMalloc(&dev_b, size * sizeof(float));
    cudaMalloc(&dev_c, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(dev_a, host_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c.data(), size * sizeof(float), cudaMemcpyHostToDevice);
  }

  void TearDown() override {
    if (dev_a)
      cudaFree(dev_a);
    if (dev_b)
      cudaFree(dev_b);
    if (dev_c)
      cudaFree(dev_c);
  }

  void CompareCudaWithCPU(const std::vector<float> &expected) {
    // Copy result back from device
    cudaMemcpy(host_c.data(), dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Compare results
    for (size_t i = 0; i < size; ++i) {
      EXPECT_NEAR(host_c[i], expected[i], 1e-6f)
          << "Mismatch at index " << i << ": GPU=" << host_c[i] << ", CPU=" << expected[i];
    }
  }

  size_t size;
  std::vector<float> host_a, host_b, host_c, host_expected;
  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
};

TEST_F(CudaKernelsTest, AddTest) {
  // Calculate expected result on CPU
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = host_a[i] + host_b[i];
  }

  // Test CUDA implementation
  cuda::cuda_add(dev_a, dev_b, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, SubTest) {
  // Calculate expected result on CPU
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = host_a[i] - host_b[i];
  }

  // Test CUDA implementation
  cuda::cuda_sub(dev_a, dev_b, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, MulTest) {
  // Calculate expected result on CPU
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = host_a[i] * host_b[i];
  }

  // Test CUDA implementation
  cuda::cuda_mul(dev_a, dev_b, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, DivTest) {
  // Calculate expected result on CPU
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = host_a[i] / host_b[i];
  }

  // Test CUDA implementation
  cuda::cuda_div(dev_a, dev_b, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, AddScalarTest) {
  float scalar = 3.14f;

  // Calculate expected result on CPU
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = host_a[i] + scalar;
  }

  // Test CUDA implementation
  cuda::cuda_add_scalar(dev_a, scalar, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, SqrtTest) {
  // Use positive values for sqrt
  for (size_t i = 0; i < size; ++i) {
    host_a[i] = static_cast<float>(i + 1);
    host_expected[i] = std::sqrt(host_a[i]);
  }

  // Copy updated data to device
  cudaMemcpy(dev_a, host_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  // Test CUDA implementation
  cuda::cuda_sqrt(dev_a, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, MaxTest) {
  // Calculate expected result on CPU
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = std::max(host_a[i], host_b[i]);
  }

  // Test CUDA implementation
  cuda::cuda_max(dev_a, dev_b, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

TEST_F(CudaKernelsTest, SumReductionTest) {
  // Calculate expected result on CPU
  float expected_sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    expected_sum += host_a[i];
  }

  // Test CUDA implementation
  float result = cuda::cuda_sum(dev_a, size, 0);
  EXPECT_NEAR(result, expected_sum, 1e-3f); // Allow for some reduction precision loss
}

TEST_F(CudaKernelsTest, DotProductTest) {
  // Calculate expected result on CPU
  float expected_dot = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    expected_dot += host_a[i] * host_b[i];
  }

  // Test CUDA implementation
  float result = cuda::cuda_dot_product(dev_a, dev_b, size, 0);
  EXPECT_NEAR(
      result, expected_dot,
      std::max(1e-3f, std::abs(expected_dot) * 1e-5f)); // Allow for some reduction precision loss
}

TEST_F(CudaKernelsTest, BatchNormOperationsTest) {
  float sub_scalar = 2.0f;
  float mul_scalar = 0.5f;

  // Test sub_mul_scalar operation
  for (size_t i = 0; i < size; ++i) {
    host_expected[i] = (host_a[i] - sub_scalar) * mul_scalar;
  }

  // Test CUDA implementation
  cuda::cuda_sub_mul_scalar(dev_a, sub_scalar, mul_scalar, dev_c, size, 0);
  CompareCudaWithCPU(host_expected);
}

// Test for double precision operations
class CudaKernelsDoubleTest : public ::testing::Test {
protected:
  void SetUp() override {
    size = 100;
    host_a.resize(size);
    host_b.resize(size);
    host_c.resize(size);

    for (size_t i = 0; i < size; ++i) {
      host_a[i] = static_cast<double>(i * 0.1);
      host_b[i] = static_cast<double>((i + 1) * 0.2);
      host_c[i] = 0.0;
    }

    cudaMalloc(&dev_a, size * sizeof(double));
    cudaMalloc(&dev_b, size * sizeof(double));
    cudaMalloc(&dev_c, size * sizeof(double));

    cudaMemcpy(dev_a, host_a.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c.data(), size * sizeof(double), cudaMemcpyHostToDevice);
  }

  void TearDown() override {
    if (dev_a)
      cudaFree(dev_a);
    if (dev_b)
      cudaFree(dev_b);
    if (dev_c)
      cudaFree(dev_c);
  }

  size_t size;
  std::vector<double> host_a, host_b, host_c;
  double *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
};

TEST_F(CudaKernelsDoubleTest, AddDoubleTest) {
  std::vector<double> expected(size);
  for (size_t i = 0; i < size; ++i) {
    expected[i] = host_a[i] + host_b[i];
  }

  cuda::cuda_add(dev_a, dev_b, dev_c, size, 0);
  cudaMemcpy(host_c.data(), dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(host_c[i], expected[i], 1e-12);
  }
}

#endif // USE_CUDA