/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "nn/layers_impl/cpu/conv2d_ops.hpp"
#include "nn/layers_impl/cuda/conv2d_ops.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

// Test fixture for CUDA conv2d operations
class CUDAConv2dOpsTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping CUDA conv2d ops tests";
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

// ==================== compute_weight_gradients Tests ====================

TEST_F(CUDAConv2dOpsTest, WeightGradientsBasic) {
  const size_t output_size = 4;
  const size_t kernel_size = 9;
  const size_t out_channels = 2;

  std::vector<float> col_data(kernel_size * output_size);
  for (size_t i = 0; i < col_data.size(); ++i) {
    col_data[i] = static_cast<float>(i) * 0.1f;
  }

  std::vector<float> gradient_data(out_channels * output_size);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i) * 0.05f;
  }

  // CPU version
  std::vector<float> cpu_weight_grad(out_channels * kernel_size, 0.0f);
  cpu::compute_weight_gradients(col_data.data(), gradient_data.data(), cpu_weight_grad.data(),
                                output_size, kernel_size, out_channels);

  // GPU version
  device_ptr<float[]> gpu_col = make_array_ptr<float[]>(gpu_device_, col_data.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight_grad =
      make_array_ptr<float[]>(gpu_device_, out_channels * kernel_size);

  gpu_device_->copyToDevice(gpu_col.get(), col_data.data(), col_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  // Initialize GPU weight grad to zero (will accumulate across micro-batches like CPU version)
  std::vector<float> zero_grad(out_channels * kernel_size, 0.0f);
  gpu_device_->copyToDevice(gpu_weight_grad.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::compute_weight_gradients(gpu_col.get(), gpu_gradient.get(), gpu_weight_grad.get(),
                                 output_size, kernel_size, out_channels);

  std::vector<float> gpu_weight_grad_cpu(out_channels * kernel_size);
  gpu_device_->copyToHost(gpu_weight_grad_cpu.data(), gpu_weight_grad.get(),
                          (out_channels * kernel_size) * sizeof(float));

  compareArrays(cpu_weight_grad, gpu_weight_grad_cpu);
}

TEST_F(CUDAConv2dOpsTest, WeightGradientsMultiOutput) {
  const size_t output_size = 9;
  const size_t kernel_size = 16; // 4x4 kernel
  const size_t out_channels = 8;

  std::vector<float> col_data(kernel_size * output_size);
  for (size_t i = 0; i < col_data.size(); ++i) {
    col_data[i] = static_cast<float>(i % 20) * 0.02f;
  }

  std::vector<float> gradient_data(out_channels * output_size);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 15) * 0.01f;
  }

  // CPU version
  std::vector<float> cpu_weight_grad(out_channels * kernel_size, 0.0f);
  cpu::compute_weight_gradients(col_data.data(), gradient_data.data(), cpu_weight_grad.data(),
                                output_size, kernel_size, out_channels);

  // GPU version
  device_ptr<float[]> gpu_col = make_array_ptr<float[]>(gpu_device_, col_data.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight_grad =
      make_array_ptr<float[]>(gpu_device_, out_channels * kernel_size);

  gpu_device_->copyToDevice(gpu_col.get(), col_data.data(), col_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(out_channels * kernel_size, 0.0f);
  gpu_device_->copyToDevice(gpu_weight_grad.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::compute_weight_gradients(gpu_col.get(), gpu_gradient.get(), gpu_weight_grad.get(),
                                 output_size, kernel_size, out_channels);

  std::vector<float> gpu_weight_grad_cpu(out_channels * kernel_size);
  gpu_device_->copyToHost(gpu_weight_grad_cpu.data(), gpu_weight_grad.get(),
                          (out_channels * kernel_size) * sizeof(float));

  compareArrays(cpu_weight_grad, gpu_weight_grad_cpu);
}

TEST_F(CUDAConv2dOpsTest, WeightGradientsLargeKernel) {
  const size_t output_size = 25;
  const size_t kernel_size = 25; // 5x5 kernel
  const size_t out_channels = 16;

  std::vector<float> col_data(kernel_size * output_size);
  for (size_t i = 0; i < col_data.size(); ++i) {
    col_data[i] = static_cast<float>(i % 50) * 0.01f;
  }

  std::vector<float> gradient_data(out_channels * output_size);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 30) * 0.005f;
  }

  // CPU version
  std::vector<float> cpu_weight_grad(out_channels * kernel_size, 0.0f);
  cpu::compute_weight_gradients(col_data.data(), gradient_data.data(), cpu_weight_grad.data(),
                                output_size, kernel_size, out_channels);

  // GPU version
  device_ptr<float[]> gpu_col = make_array_ptr<float[]>(gpu_device_, col_data.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight_grad =
      make_array_ptr<float[]>(gpu_device_, out_channels * kernel_size);

  gpu_device_->copyToDevice(gpu_col.get(), col_data.data(), col_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(out_channels * kernel_size, 0.0f);
  gpu_device_->copyToDevice(gpu_weight_grad.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  cuda::compute_weight_gradients(gpu_col.get(), gpu_gradient.get(), gpu_weight_grad.get(),
                                 output_size, kernel_size, out_channels);

  std::vector<float> gpu_weight_grad_cpu(out_channels * kernel_size);
  gpu_device_->copyToHost(gpu_weight_grad_cpu.data(), gpu_weight_grad.get(),
                          (out_channels * kernel_size) * sizeof(float));

  compareArrays(cpu_weight_grad, gpu_weight_grad_cpu);
}

// ==================== compute_input_gradients Tests ====================

TEST_F(CUDAConv2dOpsTest, InputGradientsBasic) {
  const size_t output_size = 4;
  const size_t kernel_size = 9;
  const size_t out_channels = 2;

  std::vector<float> gradient_data(out_channels * output_size);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i) * 0.05f;
  }

  std::vector<float> weight_data(out_channels * kernel_size);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_col_grad(kernel_size * output_size, 0.0f);
  cpu::compute_input_gradients(gradient_data.data(), weight_data.data(), cpu_col_grad.data(),
                               output_size, kernel_size, out_channels);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_col_grad =
      make_array_ptr<float[]>(gpu_device_, kernel_size * output_size);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  std::vector<float> zero_col_grad(kernel_size * output_size, 0.0f);
  gpu_device_->copyToDevice(gpu_col_grad.get(), zero_col_grad.data(),
                            zero_col_grad.size() * sizeof(float));

  cuda::compute_input_gradients(gpu_gradient.get(), gpu_weight.get(), gpu_col_grad.get(),
                                output_size, kernel_size, out_channels);

  std::vector<float> gpu_col_grad_cpu(kernel_size * output_size);
  gpu_device_->copyToHost(gpu_col_grad_cpu.data(), gpu_col_grad.get(),
                          (kernel_size * output_size) * sizeof(float));

  compareArrays(cpu_col_grad, gpu_col_grad_cpu);
}

TEST_F(CUDAConv2dOpsTest, InputGradientsLargeKernel) {
  const size_t output_size = 16;
  const size_t kernel_size = 25;
  const size_t out_channels = 4;

  std::vector<float> gradient_data(out_channels * output_size);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 12) * 0.03f;
  }

  std::vector<float> weight_data(out_channels * kernel_size);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i % 18) * 0.02f;
  }

  // CPU version
  std::vector<float> cpu_col_grad(kernel_size * output_size, 0.0f);
  cpu::compute_input_gradients(gradient_data.data(), weight_data.data(), cpu_col_grad.data(),
                               output_size, kernel_size, out_channels);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_col_grad =
      make_array_ptr<float[]>(gpu_device_, kernel_size * output_size);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  std::vector<float> zero_col_grad(kernel_size * output_size, 0.0f);
  gpu_device_->copyToDevice(gpu_col_grad.get(), zero_col_grad.data(),
                            zero_col_grad.size() * sizeof(float));

  cuda::compute_input_gradients(gpu_gradient.get(), gpu_weight.get(), gpu_col_grad.get(),
                                output_size, kernel_size, out_channels);

  std::vector<float> gpu_col_grad_cpu(kernel_size * output_size);
  gpu_device_->copyToHost(gpu_col_grad_cpu.data(), gpu_col_grad.get(),
                          (kernel_size * output_size) * sizeof(float));

  compareArrays(cpu_col_grad, gpu_col_grad_cpu);
}

// ==================== compute_bias_gradients Tests ====================

TEST_F(CUDAConv2dOpsTest, BiasGradientsBasic) {
  const size_t batch_size = 2;
  const size_t output_h = 3;
  const size_t output_w = 3;
  const size_t out_channels = 2;

  std::vector<float> gradient_data(batch_size * out_channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i) * 0.01f;
  }

  // CPU version
  std::vector<float> cpu_bias_grad(out_channels, 0.0f);
  cpu::compute_bias_gradients(gradient_data.data(), cpu_bias_grad.data(), batch_size, output_h,
                              output_w, out_channels);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_bias_grad = make_array_ptr<float[]>(gpu_device_, out_channels);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_bias_grad(out_channels, 0.0f);
  gpu_device_->copyToDevice(gpu_bias_grad.get(), zero_bias_grad.data(),
                            out_channels * sizeof(float));

  cuda::compute_bias_gradients(gpu_gradient.get(), gpu_bias_grad.get(), batch_size, output_h,
                               output_w, out_channels);

  std::vector<float> gpu_bias_grad_cpu(out_channels);
  gpu_device_->copyToHost(gpu_bias_grad_cpu.data(), gpu_bias_grad.get(),
                          out_channels * sizeof(float));

  compareArrays(cpu_bias_grad, gpu_bias_grad_cpu);
}

TEST_F(CUDAConv2dOpsTest, BiasGradientsMultiBatch) {
  const size_t batch_size = 4;
  const size_t output_h = 5;
  const size_t output_w = 5;
  const size_t out_channels = 16;

  std::vector<float> gradient_data(batch_size * out_channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 100) * 0.001f;
  }

  // CPU version
  std::vector<float> cpu_bias_grad(out_channels, 0.0f);
  cpu::compute_bias_gradients(gradient_data.data(), cpu_bias_grad.data(), batch_size, output_h,
                              output_w, out_channels);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_bias_grad = make_array_ptr<float[]>(gpu_device_, out_channels);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_bias_grad(out_channels, 0.0f);
  gpu_device_->copyToDevice(gpu_bias_grad.get(), zero_bias_grad.data(),
                            out_channels * sizeof(float));

  cuda::compute_bias_gradients(gpu_gradient.get(), gpu_bias_grad.get(), batch_size, output_h,
                               output_w, out_channels);

  std::vector<float> gpu_bias_grad_cpu(out_channels);
  gpu_device_->copyToHost(gpu_bias_grad_cpu.data(), gpu_bias_grad.get(),
                          out_channels * sizeof(float));

  compareArrays(cpu_bias_grad, gpu_bias_grad_cpu);
}

TEST_F(CUDAConv2dOpsTest, BiasGradientsLargeChannels) {
  const size_t batch_size = 8;
  const size_t output_h = 8;
  const size_t output_w = 8;
  const size_t out_channels = 64;

  std::vector<float> gradient_data(batch_size * out_channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 256) * 0.0001f;
  }

  // CPU version
  std::vector<float> cpu_bias_grad(out_channels, 0.0f);
  cpu::compute_bias_gradients(gradient_data.data(), cpu_bias_grad.data(), batch_size, output_h,
                              output_w, out_channels);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_bias_grad = make_array_ptr<float[]>(gpu_device_, out_channels);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_bias_grad(out_channels, 0.0f);
  gpu_device_->copyToDevice(gpu_bias_grad.get(), zero_bias_grad.data(),
                            out_channels * sizeof(float));

  cuda::compute_bias_gradients(gpu_gradient.get(), gpu_bias_grad.get(), batch_size, output_h,
                               output_w, out_channels);

  std::vector<float> gpu_bias_grad_cpu(out_channels);
  gpu_device_->copyToHost(gpu_bias_grad_cpu.data(), gpu_bias_grad.get(),
                          out_channels * sizeof(float));

  compareArrays(cpu_bias_grad, gpu_bias_grad_cpu);
}

// ==================== add_bias_to_output Tests ====================

TEST_F(CUDAConv2dOpsTest, AddBiasBasic) {
  const size_t batch_size = 1;
  const size_t output_h = 3;
  const size_t output_w = 3;
  const size_t out_channels = 2;

  std::vector<float> output_data(batch_size * out_channels * output_h * output_w);
  for (size_t i = 0; i < output_data.size(); ++i) {
    output_data[i] = static_cast<float>(i) * 0.1f;
  }

  std::vector<float> bias_data(out_channels);
  for (size_t i = 0; i < bias_data.size(); ++i) {
    bias_data[i] = static_cast<float>(i + 1) * 0.5f;
  }

  // CPU version
  std::vector<float> cpu_output = output_data;
  cpu::add_bias_to_output(cpu_output.data(), bias_data.data(), batch_size, output_h, output_w,
                          out_channels);

  // GPU version
  device_ptr<float[]> gpu_output = make_array_ptr<float[]>(gpu_device_, output_data.size());
  device_ptr<float[]> gpu_bias = make_array_ptr<float[]>(gpu_device_, bias_data.size());

  gpu_device_->copyToDevice(gpu_output.get(), output_data.data(),
                            output_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_bias.get(), bias_data.data(), bias_data.size() * sizeof(float));

  cuda::add_bias_to_output(gpu_output.get(), gpu_bias.get(), batch_size, output_h, output_w,
                           out_channels);

  std::vector<float> gpu_output_cpu(output_data.size());
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          output_data.size() * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

TEST_F(CUDAConv2dOpsTest, AddBiasMultiBatch) {
  const size_t batch_size = 4;
  const size_t output_h = 4;
  const size_t output_w = 4;
  const size_t out_channels = 8;

  std::vector<float> output_data(batch_size * out_channels * output_h * output_w);
  for (size_t i = 0; i < output_data.size(); ++i) {
    output_data[i] = static_cast<float>(i % 20) * 0.05f;
  }

  std::vector<float> bias_data(out_channels);
  for (size_t i = 0; i < bias_data.size(); ++i) {
    bias_data[i] = static_cast<float>(i) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_output = output_data;
  cpu::add_bias_to_output(cpu_output.data(), bias_data.data(), batch_size, output_h, output_w,
                          out_channels);

  // GPU version
  device_ptr<float[]> gpu_output = make_array_ptr<float[]>(gpu_device_, output_data.size());
  device_ptr<float[]> gpu_bias = make_array_ptr<float[]>(gpu_device_, bias_data.size());

  gpu_device_->copyToDevice(gpu_output.get(), output_data.data(),
                            output_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_bias.get(), bias_data.data(), bias_data.size() * sizeof(float));

  cuda::add_bias_to_output(gpu_output.get(), gpu_bias.get(), batch_size, output_h, output_w,
                           out_channels);

  std::vector<float> gpu_output_cpu(output_data.size());
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          output_data.size() * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

TEST_F(CUDAConv2dOpsTest, AddBiasLargeOutput) {
  const size_t batch_size = 8;
  const size_t output_h = 16;
  const size_t output_w = 16;
  const size_t out_channels = 32;

  std::vector<float> output_data(batch_size * out_channels * output_h * output_w);
  for (size_t i = 0; i < output_data.size(); ++i) {
    output_data[i] = static_cast<float>(i % 100) * 0.001f;
  }

  std::vector<float> bias_data(out_channels);
  for (size_t i = 0; i < bias_data.size(); ++i) {
    bias_data[i] = static_cast<float>(i) * 0.01f;
  }

  // CPU version
  std::vector<float> cpu_output = output_data;
  cpu::add_bias_to_output(cpu_output.data(), bias_data.data(), batch_size, output_h, output_w,
                          out_channels);

  // GPU version
  device_ptr<float[]> gpu_output = make_array_ptr<float[]>(gpu_device_, output_data.size());
  device_ptr<float[]> gpu_bias = make_array_ptr<float[]>(gpu_device_, bias_data.size());

  gpu_device_->copyToDevice(gpu_output.get(), output_data.data(),
                            output_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_bias.get(), bias_data.data(), bias_data.size() * sizeof(float));

  cuda::add_bias_to_output(gpu_output.get(), gpu_bias.get(), batch_size, output_h, output_w,
                           out_channels);

  std::vector<float> gpu_output_cpu(output_data.size());
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          output_data.size() * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
