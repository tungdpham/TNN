/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "device/task.hpp"
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
      if (device.device_type() == DeviceType::GPU) {
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

  // Helper function to compare mask indices from device_ptrs
  void compareMasks(const device_ptr<size_t[]> &expected, const device_ptr<size_t[]> &actual,
                    size_t size) {
    ASSERT_EQ(expected.size(), actual.size())
        << "Mask sizes don't match: expected " << expected.size() << ", got " << actual.size();

    // Copy masks to host for comparison
    std::vector<size_t> expected_host(size);
    std::vector<size_t> actual_host(size);

    if (expected.device_type() == DeviceType::CPU) {
      std::memcpy(expected_host.data(), expected.get(), size * sizeof(size_t));
    } else {
      expected.getDevice()->copyToHost(expected_host.data(), expected.get(), size * sizeof(size_t));
    }

    if (actual.device_type() == DeviceType::CPU) {
      std::memcpy(actual_host.data(), actual.get(), size * sizeof(size_t));
    } else {
      actual.getDevice()->copyToHost(actual_host.data(), actual.get(), size * sizeof(size_t));
    }

    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(expected_host[i], actual_host[i]) << "Mask mismatch at index " << i << ": expected "
                                                  << expected_host[i] << ", got " << actual_host[i];
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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), cpu_output.data(), batch_size, channels,
                                         input_h, input_w, output_h, output_w, pool_h, pool_w,
                                         stride_h, stride_w, cpu_mask);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);
  auto gpu_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_output.get(), batch_size, channels, input_h, input_w,
                      output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), cpu_output.data(), batch_size, channels,
                                         input_h, input_w, output_h, output_w, pool_h, pool_w,
                                         stride_h, stride_w, cpu_mask);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);
  auto gpu_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_output.get(), batch_size, channels, input_h, input_w,
                      output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  auto cpu_task =
      create_cpu_task("test_maxpool_forward_cpu", cpu::maxpool::compute_max_pool_forward<float>,
                      input_data.data(), cpu_output.data(), batch_size, channels, input_h, input_w,
                      output_h, output_w, pool_h, pool_w, stride_h, stride_w, cpu_mask);
  ASSERT_FALSE(cpu_task->sync()) << "CPU maxpool forward task failed";

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);
  auto gpu_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_output.get(), batch_size, channels, input_h, input_w,
                      output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // CPU version
  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  auto cpu_task =
      create_cpu_task("test_maxpool_forward_cpu", cpu::maxpool::compute_max_pool_forward<float>,
                      input_data.data(), cpu_output.data(), batch_size, channels, input_h, input_w,
                      output_h, output_w, pool_h, pool_w, stride_h, stride_w, cpu_mask);
  ASSERT_FALSE(cpu_task->sync()) << "CPU maxpool forward task failed";

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * output_h * output_w);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);
  auto gpu_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_output.get(), batch_size, channels, input_h, input_w,
                      output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // First do forward pass to get mask on CPU
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  cpu::maxpool::compute_max_pool_forward(input_data.data(), forward_output.data(), batch_size,
                                         channels, input_h, input_w, output_h, output_w, pool_h,
                                         pool_w, stride_h, stride_w, cpu_mask);

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  cpu::maxpool::compute_max_pool_backward(gradient_data.data(), cpu_grad_input.data(), batch_size,
                                          channels, output_h, output_w, cpu_mask);

  // GPU version - need to do forward pass on GPU to get GPU mask
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_forward_output = make_array_ptr<float[]>(gpu_device_, mask_size);
  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_forward_output.get(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_backward_task = create_gpu_task(
      "test_maxpool_backward_gpu", cuda::maxpool::compute_max_pool_backward<float>,
      gpu_gradient.get(), gpu_grad_input.get(), batch_size, channels, output_h, output_w, gpu_mask);
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // First do forward pass to get mask on CPU
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  auto cpu_forward_task =
      create_cpu_task("test_maxpool_forward_cpu", cpu::maxpool::compute_max_pool_forward<float>,
                      input_data.data(), forward_output.data(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, cpu_mask);
  ASSERT_FALSE(cpu_forward_task->sync()) << "CPU maxpool forward task failed";

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>((i % 50) + 1) * 0.05f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  auto cpu_backward_task =
      create_cpu_task("test_maxpool_backward_cpu", cpu::maxpool::compute_max_pool_backward<float>,
                      gradient_data.data(), cpu_grad_input.data(), batch_size, channels, output_h,
                      output_w, cpu_mask);
  ASSERT_FALSE(cpu_backward_task->sync()) << "CPU maxpool backward task failed";

  // GPU version - need to do forward pass on GPU to get GPU mask
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_forward_output = make_array_ptr<float[]>(gpu_device_, mask_size);
  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_forward_output.get(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_backward_task = create_gpu_task(
      "test_maxpool_backward_gpu", cuda::maxpool::compute_max_pool_backward<float>,
      gpu_gradient.get(), gpu_grad_input.get(), batch_size, channels, output_h, output_w, gpu_mask);
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // First do forward pass to get mask on CPU
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  auto cpu_forward_task =
      create_cpu_task("test_maxpool_forward_cpu", cpu::maxpool::compute_max_pool_forward<float>,
                      input_data.data(), forward_output.data(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, cpu_mask);
  ASSERT_FALSE(cpu_forward_task->sync()) << "CPU maxpool forward task failed";

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>((i + 1)) * 0.2f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  auto cpu_backward_task =
      create_cpu_task("test_maxpool_backward_cpu", cpu::maxpool::compute_max_pool_backward<float>,
                      gradient_data.data(), cpu_grad_input.data(), batch_size, channels, output_h,
                      output_w, cpu_mask);
  ASSERT_FALSE(cpu_backward_task->sync()) << "CPU maxpool backward task failed";

  // GPU version - need to do forward pass on GPU to get GPU mask
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_forward_output = make_array_ptr<float[]>(gpu_device_, mask_size);
  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_forward_output.get(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_backward_task = create_gpu_task(
      "test_maxpool_backward_gpu", cuda::maxpool::compute_max_pool_backward<float>,
      gpu_gradient.get(), gpu_grad_input.get(), batch_size, channels, output_h, output_w, gpu_mask);
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

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

  const Device &cpu_device = getCPU();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  // First do forward pass to get mask on CPU
  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  device_ptr<size_t[]> cpu_mask = make_array_ptr<size_t[]>(&cpu_device, mask_size);
  auto cpu_forward_task =
      create_cpu_task("test_maxpool_forward_cpu", cpu::maxpool::compute_max_pool_forward<float>,
                      input_data.data(), forward_output.data(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, cpu_mask);
  ASSERT_FALSE(cpu_forward_task->sync()) << "CPU maxpool forward task failed";

  // Create gradient for backward pass
  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.15f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  auto cpu_backward_task =
      create_cpu_task("test_maxpool_backward_cpu", cpu::maxpool::compute_max_pool_backward<float>,
                      gradient_data.data(), cpu_grad_input.data(), batch_size, channels, output_h,
                      output_w, cpu_mask);
  ASSERT_FALSE(cpu_backward_task->sync()) << "CPU maxpool backward task failed";

  // GPU version - need to do forward pass on GPU to get GPU mask
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_forward_output = make_array_ptr<float[]>(gpu_device_, mask_size);
  device_ptr<size_t[]> gpu_mask = make_array_ptr<size_t[]>(gpu_device_, mask_size);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_gpu_task("test_maxpool_forward_gpu", cuda::maxpool::compute_max_pool_forward<float>,
                      gpu_input.get(), gpu_forward_output.get(), batch_size, channels, input_h,
                      input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w, gpu_mask);
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * input_h * input_w);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_backward_task = create_gpu_task(
      "test_maxpool_backward_gpu", cuda::maxpool::compute_max_pool_backward<float>,
      gpu_gradient.get(), gpu_grad_input.get(), batch_size, channels, output_h, output_w, gpu_mask);
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

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
