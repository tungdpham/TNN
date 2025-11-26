/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/batchnorm_ops.hpp"
#include "nn/layers_impl/cuda/batchnorm_ops.hpp"
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

#ifdef USE_CUDA
// Test fixture for CUDA batchnorm operations
class CUDABatchNormOpsTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping CUDA batchnorm ops tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  // Helper function to compare arrays with tolerance
  void compareArrays(const std::vector<float> &expected, const std::vector<float> &actual,
                     float tolerance = 1e-4f) {
    ASSERT_EQ(expected.size(), actual.size())
        << "Arrays have different sizes. Expected: " << expected.size()
        << ", Actual: " << actual.size();

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << "Mismatch at index " << i << ". Expected: " << expected[i] << ", Got: " << actual[i];
    }
  }

  bool has_gpu_;
  const Device *gpu_device_;
};

// ==================== compute_inference_output Tests ====================

TEST_F(CUDABatchNormOpsTest, InferenceOutputAffine) {
  const size_t batch_size = 2;
  const size_t channels = 2;
  const size_t spatial_size = 4;
  const size_t total_size = batch_size * channels * spatial_size;
  const float epsilon = 1e-5f;
  const bool affine = true;

  std::vector<float> input_data(total_size);
  for (size_t i = 0; i < total_size; ++i)
    input_data[i] = static_cast<float>(i) * 0.1f;

  std::vector<float> running_mean(channels);
  std::vector<float> running_var(channels);
  std::vector<float> gamma(channels);
  std::vector<float> beta(channels);

  for (size_t i = 0; i < channels; ++i) {
    running_mean[i] = static_cast<float>(i) * 0.1f;
    running_var[i] = 1.0f + static_cast<float>(i) * 0.2f;
    gamma[i] = 0.9f + static_cast<float>(i) * 0.05f;
    beta[i] = -0.2f + static_cast<float>(i) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_output(total_size);
  cpu::batchnorm::compute_inference_output(
      input_data.data(), running_mean.data(), running_var.data(), gamma.data(), beta.data(),
      cpu_output.data(), batch_size, channels, spatial_size, epsilon, affine);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, total_size);
  device_ptr<float[]> gpu_running_mean = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_running_var = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_gamma = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_beta = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_output = make_array_ptr<float[]>(gpu_device_, total_size);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_running_mean.get(), running_mean.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_running_var.get(), running_var.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_gamma.get(), gamma.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_beta.get(), beta.data(), channels * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_inference_gpu", cuda::batchnorm::compute_inference_output<float>, gpu_input.get(),
      gpu_running_mean.get(), gpu_running_var.get(), gpu_gamma.get(), gpu_beta.get(),
      gpu_output.get(), batch_size, channels, spatial_size, epsilon, affine);
  ASSERT_FALSE(gpu_task->sync());

  std::vector<float> gpu_output_cpu(total_size);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(), total_size * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

// ==================== compute_batchnorm_backward_fused Tests ====================

TEST_F(CUDABatchNormOpsTest, BackwardFusedAffine) {
  const size_t batch_size = 2;
  const size_t channels = 2;
  const size_t spatial_size = 4;
  const size_t total_size = batch_size * channels * spatial_size;
  const bool affine = true;

  std::vector<float> gradient_data(total_size);
  std::vector<float> normalized_data(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    gradient_data[i] = static_cast<float>(i) * 0.01f;
    normalized_data[i] = static_cast<float>(i % 10) * 0.1f - 0.5f;
  }

  std::vector<float> std_data(channels);
  std::vector<float> gamma_data(channels);
  for (size_t i = 0; i < channels; ++i) {
    std_data[i] = 1.0f + static_cast<float>(i) * 0.1f;
    gamma_data[i] = 0.8f + static_cast<float>(i) * 0.05f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(total_size);
  std::vector<float> cpu_gamma_grad(channels);
  std::vector<float> cpu_beta_grad(channels);

  cpu::batchnorm::run_backward_fused(gradient_data.data(), normalized_data.data(), std_data.data(),
                                     gamma_data.data(), cpu_gamma_grad.data(), cpu_beta_grad.data(),
                                     cpu_grad_input.data(), batch_size, channels, spatial_size,
                                     affine);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, total_size);
  device_ptr<float[]> gpu_normalized = make_array_ptr<float[]>(gpu_device_, total_size);
  device_ptr<float[]> gpu_std = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_gamma = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_grad_input = make_array_ptr<float[]>(gpu_device_, total_size);
  device_ptr<float[]> gpu_gamma_grad = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_beta_grad = make_array_ptr<float[]>(gpu_device_, channels);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(), total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_normalized.get(), normalized_data.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_std.get(), std_data.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_gamma.get(), gamma_data.data(), channels * sizeof(float));

  // Initialize outputs with zeros
  std::vector<float> zeros_total(total_size, 0.0f);
  std::vector<float> zeros_channels(channels, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zeros_total.data(), total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_gamma_grad.get(), zeros_channels.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_beta_grad.get(), zeros_channels.data(), channels * sizeof(float));

  auto gpu_task = create_gpu_task("test_backward_gpu", cuda::batchnorm::run_backward_fused<float>,
                                  gpu_gradient.get(), gpu_normalized.get(), gpu_std.get(),
                                  gpu_gamma.get(), gpu_gamma_grad.get(), gpu_beta_grad.get(),
                                  gpu_grad_input.get(), batch_size, channels, spatial_size, affine);
  ASSERT_FALSE(gpu_task->sync());

  std::vector<float> gpu_grad_input_cpu(total_size);
  std::vector<float> gpu_gamma_grad_cpu(channels);
  std::vector<float> gpu_beta_grad_cpu(channels);

  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          total_size * sizeof(float));
  gpu_device_->copyToHost(gpu_gamma_grad_cpu.data(), gpu_gamma_grad.get(),
                          channels * sizeof(float));
  gpu_device_->copyToHost(gpu_beta_grad_cpu.data(), gpu_beta_grad.get(), channels * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
  compareArrays(cpu_gamma_grad, gpu_gamma_grad_cpu);
  compareArrays(cpu_beta_grad, gpu_beta_grad_cpu);
}

TEST_F(CUDABatchNormOpsTest, BackwardFusedNoAffine) {
  const size_t batch_size = 2;
  const size_t channels = 2;
  const size_t spatial_size = 4;
  const size_t total_size = batch_size * channels * spatial_size;
  const bool affine = false;

  std::vector<float> gradient_data(total_size);
  std::vector<float> normalized_data(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    gradient_data[i] = static_cast<float>(i) * 0.01f;
    normalized_data[i] = static_cast<float>(i % 10) * 0.1f - 0.5f;
  }

  std::vector<float> inv_std_data(channels);
  for (size_t i = 0; i < channels; ++i) {
    inv_std_data[i] = 1.0f + static_cast<float>(i) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(total_size);
  std::vector<float> cpu_gamma_grad(channels, 0.0f);   // Not used
  std::vector<float> cpu_d_gamma_grad(channels, 0.0f); // Not used
  std::vector<float> cpu_d_beta_grad(channels, 0.0f);

  cpu::batchnorm::run_backward_fused(
      gradient_data.data(), normalized_data.data(), inv_std_data.data(), cpu_gamma_grad.data(),
      cpu_d_gamma_grad.data(), cpu_d_beta_grad.data(), cpu_grad_input.data(), batch_size, channels,
      spatial_size, affine);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, total_size);
  device_ptr<float[]> gpu_normalized = make_array_ptr<float[]>(gpu_device_, total_size);
  device_ptr<float[]> gpu_inv_std = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_grad_input = make_array_ptr<float[]>(gpu_device_, total_size);

  // Allocate dummy buffers for gamma, d_gamma and d_beta even though affine=false
  // The CUDA kernels need these for intermediate computations
  device_ptr<float[]> gpu_dummy_gamma = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_dummy_gamma_grad = make_array_ptr<float[]>(gpu_device_, channels);
  device_ptr<float[]> gpu_dummy_beta_grad = make_array_ptr<float[]>(gpu_device_, channels);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(), total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_normalized.get(), normalized_data.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_inv_std.get(), inv_std_data.data(), channels * sizeof(float));

  std::vector<float> zeros_total(total_size, 0.0f);
  std::vector<float> zeros_channels(channels, 0.0f);
  std::vector<float> ones_channels(channels, 1.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zeros_total.data(), total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_dummy_gamma.get(), ones_channels.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_dummy_gamma_grad.get(), zeros_channels.data(),
                            channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_dummy_beta_grad.get(), zeros_channels.data(),
                            channels * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_backward_gpu", cuda::batchnorm::run_backward_fused<float>, gpu_gradient.get(),
      gpu_normalized.get(), gpu_inv_std.get(), gpu_dummy_gamma.get(), gpu_dummy_gamma_grad.get(),
      gpu_dummy_beta_grad.get(), gpu_grad_input.get(), batch_size, channels, spatial_size, affine);
  ASSERT_FALSE(gpu_task->sync());

  std::vector<float> gpu_grad_input_cpu(total_size);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          total_size * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // USE_CUDA
