/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/batchnorm_nchw_ops.hpp"
#include "nn/layers_impl/cuda/batchnorm_nchw_ops.hpp"
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

#ifdef USE_CUDA

class CUDABatchNormOpsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

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

  std::vector<float> cpu_output(total_size);
  cpu::batchnorm_nchw::compute_inference_output(
      input_data.data(), running_mean.data(), running_var.data(), gamma.data(), beta.data(),
      cpu_output.data(), batch_size, channels, spatial_size, epsilon, affine);

  device_ptr gpu_input = make_dptr_t<float[]>(gpu_device_, total_size);
  device_ptr gpu_running_mean = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_running_var = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_gamma = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_beta = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_output = make_dptr_t<float[]>(gpu_device_, total_size);

  gpu_device_->copyToDevice(gpu_input.get<float>(), input_data.data(), total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_running_mean.get<float>(), running_mean.data(),
                            channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_running_var.get<float>(), running_var.data(),
                            channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_gamma.get<float>(), gamma.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_beta.get<float>(), beta.data(), channels * sizeof(float));

  auto gpu_task =
      create_gpu_task("test_inference_gpu", cuda::batchnorm_nchw::compute_inference_output<float>,
                      gpu_input.get<float>(), gpu_running_mean.get<float>(),
                      gpu_running_var.get<float>(), gpu_gamma.get<float>(), gpu_beta.get<float>(),
                      gpu_output.get<float>(), batch_size, channels, spatial_size, epsilon, affine);
  ASSERT_FALSE(gpu_task->sync()) << "GPU batchnorm inference task failed";

  std::vector<float> gpu_output_cpu(total_size);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get<float>(),
                          total_size * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

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

  std::vector<float> cpu_grad_input(total_size);
  std::vector<float> cpu_gamma_grad(channels);
  std::vector<float> cpu_beta_grad(channels);

  cpu::batchnorm_nchw::run_backward_fused(gradient_data.data(), normalized_data.data(),
                                          std_data.data(), gamma_data.data(), cpu_gamma_grad.data(),
                                          cpu_beta_grad.data(), cpu_grad_input.data(), batch_size,
                                          channels, spatial_size, affine);

  device_ptr gpu_gradient = make_dptr_t<float[]>(gpu_device_, total_size);
  device_ptr gpu_normalized = make_dptr_t<float[]>(gpu_device_, total_size);
  device_ptr gpu_std = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_gamma = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_grad_input = make_dptr_t<float[]>(gpu_device_, total_size);
  device_ptr gpu_gamma_grad = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_beta_grad = make_dptr_t<float[]>(gpu_device_, channels);

  gpu_device_->copyToDevice(gpu_gradient.get<float>(), gradient_data.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_normalized.get<float>(), normalized_data.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_std.get<float>(), std_data.data(), channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_gamma.get<float>(), gamma_data.data(), channels * sizeof(float));

  std::vector<float> zeros_total(total_size, 0.0f);
  std::vector<float> zeros_channels(channels, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get<float>(), zeros_total.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_gamma_grad.get<float>(), zeros_channels.data(),
                            channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_beta_grad.get<float>(), zeros_channels.data(),
                            channels * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_backward_gpu", cuda::batchnorm_nchw::run_backward_fused<float>,
      gpu_gradient.get<float>(), gpu_normalized.get<float>(), gpu_std.get<float>(),
      gpu_gamma.get<float>(), gpu_gamma_grad.get<float>(), gpu_beta_grad.get<float>(),
      gpu_grad_input.get<float>(), batch_size, channels, spatial_size, affine);
  ASSERT_FALSE(gpu_task->sync()) << "GPU batchnorm backward task failed";

  std::vector<float> gpu_grad_input_cpu(total_size);
  std::vector<float> gpu_gamma_grad_cpu(channels);
  std::vector<float> gpu_beta_grad_cpu(channels);

  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get<float>(),
                          total_size * sizeof(float));
  gpu_device_->copyToHost(gpu_gamma_grad_cpu.data(), gpu_gamma_grad.get<float>(),
                          channels * sizeof(float));
  gpu_device_->copyToHost(gpu_beta_grad_cpu.data(), gpu_beta_grad.get<float>(),
                          channels * sizeof(float));

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

  std::vector<float> cpu_grad_input(total_size);
  std::vector<float> cpu_gamma_grad(channels, 0.0f);
  std::vector<float> cpu_d_gamma_grad(channels, 0.0f);
  std::vector<float> cpu_d_beta_grad(channels, 0.0f);

  cpu::batchnorm_nchw::run_backward_fused(
      gradient_data.data(), normalized_data.data(), inv_std_data.data(), cpu_gamma_grad.data(),
      cpu_d_gamma_grad.data(), cpu_d_beta_grad.data(), cpu_grad_input.data(), batch_size, channels,
      spatial_size, affine);

  device_ptr gpu_gradient = make_dptr_t<float[]>(gpu_device_, total_size);
  device_ptr gpu_normalized = make_dptr_t<float[]>(gpu_device_, total_size);
  device_ptr gpu_inv_std = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_grad_input = make_dptr_t<float[]>(gpu_device_, total_size);

  device_ptr gpu_dummy_gamma = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_dummy_gamma_grad = make_dptr_t<float[]>(gpu_device_, channels);
  device_ptr gpu_dummy_beta_grad = make_dptr_t<float[]>(gpu_device_, channels);

  gpu_device_->copyToDevice(gpu_gradient.get<float>(), gradient_data.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_normalized.get<float>(), normalized_data.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_inv_std.get<float>(), inv_std_data.data(),
                            channels * sizeof(float));

  std::vector<float> zeros_total(total_size, 0.0f);
  std::vector<float> zeros_channels(channels, 0.0f);
  std::vector<float> ones_channels(channels, 1.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get<float>(), zeros_total.data(),
                            total_size * sizeof(float));
  gpu_device_->copyToDevice(gpu_dummy_gamma.get<float>(), ones_channels.data(),
                            channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_dummy_gamma_grad.get<float>(), zeros_channels.data(),
                            channels * sizeof(float));
  gpu_device_->copyToDevice(gpu_dummy_beta_grad.get<float>(), zeros_channels.data(),
                            channels * sizeof(float));

  auto gpu_task =
      create_gpu_task("test_backward_gpu", cuda::batchnorm_nchw::run_backward_fused<float>,
                      gpu_gradient.get<float>(), gpu_normalized.get<float>(),
                      gpu_inv_std.get<float>(), gpu_dummy_gamma.get<float>(),
                      gpu_dummy_gamma_grad.get<float>(), gpu_dummy_beta_grad.get<float>(),
                      gpu_grad_input.get<float>(), batch_size, channels, spatial_size, affine);
  ASSERT_FALSE(gpu_task->sync()) << "GPU batchnorm backward task failed";

  std::vector<float> gpu_grad_input_cpu(total_size);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get<float>(),
                          total_size * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
