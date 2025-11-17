/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "nn/layers_impl/cpu/maxpool_ops.hpp"
#include "nn/layers_impl/cuda/maxpool_ops.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

#ifdef USE_CUDA
// Test fixture for CUDA maxpool operations
class CUDAMaxPoolOpsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Initialize devices once for all tests in this suite
    initializeDefaultDevices();
  }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    // Find GPU device
    has_gpu_ = false;
    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.getDeviceType() == DeviceType::GPU) {
        gpu_device_ = &device;
        has_gpu_ = true;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No GPU device available, skipping CUDA maxpool ops tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  // Helper function to compare arrays with tolerance
  void compareArrays(const std::vector<float> &expected, const std::vector<float> &actual,
                     float tolerance = 1e-4f) {
    ASSERT_EQ(expected.size(), actual.size())
        << "Array sizes don't match: expected " << expected.size() << ", got " << actual.size();

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i];
    }
  }

  // Helper function to compare mask indices
  void compareMasks(const std::vector<size_t> &expected, const std::vector<size_t> &actual) {
    ASSERT_EQ(expected.size(), actual.size())
        << "Mask sizes don't match: expected " << expected.size() << ", got " << actual.size();

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], actual[i]) << "Mask mismatch at index " << i << ": expected "
                                        << expected[i] << ", got " << actual[i];
    }
  }

  bool has_gpu_;
  const Device *gpu_device_;
};

// ==================== compute_max_pool_forward Tests ====================

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardBasic) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 4;
  const size_t input_w = 4;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> cpu_mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), cpu_output.data(), batch_size, channels,
                                         input_h, input_w, output_h, output_w, pool_h, pool_w,
                                         stride_h, stride_w, cpu_mask);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  std::vector<size_t> gpu_mask(batch_size * channels * output_h * output_w);
  cuda::maxpool::compute_max_pool_forward(gpu_input.get(), gpu_output.get(), batch_size, channels,
                                          input_h, input_w, output_h, output_w, pool_h, pool_w,
                                          stride_h, stride_w, gpu_mask);

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardMultiChannel) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 6;
  const size_t input_w = 6;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> cpu_mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), cpu_output.data(), batch_size, channels,
                                         input_h, input_w, output_h, output_w, pool_h, pool_w,
                                         stride_h, stride_w, cpu_mask);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  std::vector<size_t> gpu_mask(batch_size * channels * output_h * output_w);
  cuda::maxpool::compute_max_pool_forward(gpu_input.get(), gpu_output.get(), batch_size, channels,
                                          input_h, input_w, output_h, output_w, pool_h, pool_w,
                                          stride_h, stride_w, gpu_mask);

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardLargePool) {
  const size_t batch_size = 1;
  const size_t channels = 2;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 3;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i * 7) % 100) * 0.1f; // Create varied pattern
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> cpu_mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), cpu_output.data(), batch_size, channels,
                                         input_h, input_w, output_h, output_w, pool_h, pool_w,
                                         stride_h, stride_w, cpu_mask);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  std::vector<size_t> gpu_mask(batch_size * channels * output_h * output_w);
  cuda::maxpool::compute_max_pool_forward(gpu_input.get(), gpu_output.get(), batch_size, channels,
                                          input_h, input_w, output_h, output_w, pool_h, pool_w,
                                          stride_h, stride_w, gpu_mask);

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardNonSquare) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 8;
  const size_t input_w = 12;
  const size_t pool_h = 2;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 3;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> cpu_mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), cpu_output.data(), batch_size, channels,
                                         input_h, input_w, output_h, output_w, pool_h, pool_w,
                                         stride_h, stride_w, cpu_mask);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  std::vector<size_t> gpu_mask(batch_size * channels * output_h * output_w);
  cuda::maxpool::compute_max_pool_forward(gpu_input.get(), gpu_output.get(), batch_size, channels,
                                          input_h, input_w, output_h, output_w, pool_h, pool_w,
                                          stride_h, stride_w, gpu_mask);

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask);
}

// ==================== compute_max_pool_backward Tests ====================

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardBasic) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 4;
  const size_t input_w = 4;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  // First do forward pass to get mask
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), forward_output.data(), batch_size,
                                         channels, input_h, input_w, output_h, output_w, pool_h,
                                         pool_w, stride_h, stride_w, mask);

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  cpu::maxpool::compute_max_pool_backward(gradient_data.data(), cpu_grad_input.data(), batch_size,
                                          channels, output_h, output_w, mask);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::maxpool::compute_max_pool_backward(gpu_gradient.get(), gpu_grad_input.get(), batch_size,
                                           channels, output_h, output_w, mask);

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardMultiChannel) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 6;
  const size_t input_w = 6;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) + 1) * 0.1f;
  }

  // First do forward pass to get mask
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), forward_output.data(), batch_size,
                                         channels, input_h, input_w, output_h, output_w, pool_h,
                                         pool_w, stride_h, stride_w, mask);

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>((i % 50) + 1) * 0.05f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  cpu::maxpool::compute_max_pool_backward(gradient_data.data(), cpu_grad_input.data(), batch_size,
                                          channels, output_h, output_w, mask);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::maxpool::compute_max_pool_backward(gpu_gradient.get(), gpu_grad_input.get(), batch_size,
                                           channels, output_h, output_w, mask);

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardLargePool) {
  const size_t batch_size = 1;
  const size_t channels = 2;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 3;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i * 7) % 100) * 0.1f;
  }

  // First do forward pass to get mask
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), forward_output.data(), batch_size,
                                         channels, input_h, input_w, output_h, output_w, pool_h,
                                         pool_w, stride_h, stride_w, mask);

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>((i + 1)) * 0.2f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  cpu::maxpool::compute_max_pool_backward(gradient_data.data(), cpu_grad_input.data(), batch_size,
                                          channels, output_h, output_w, mask);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::maxpool::compute_max_pool_backward(gpu_gradient.get(), gpu_grad_input.get(), batch_size,
                                           channels, output_h, output_w, mask);

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardNonSquare) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 8;
  const size_t input_w = 12;
  const size_t pool_h = 2;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 3;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  // First do forward pass to get mask
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  std::vector<size_t> mask(batch_size * channels * output_h * output_w);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), forward_output.data(), batch_size,
                                         channels, input_h, input_w, output_h, output_w, pool_h,
                                         pool_w, stride_h, stride_w, mask);

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.15f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  cpu::maxpool::compute_max_pool_backward(gradient_data.data(), cpu_grad_input.data(), batch_size,
                                          channels, output_h, output_w, mask);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::maxpool::compute_max_pool_backward(gpu_gradient.get(), gpu_grad_input.get(), batch_size,
                                           channels, output_h, output_w, mask);

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // USE_CUDA
