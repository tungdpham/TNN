/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/dense_ops.hpp"
#include "nn/layers_impl/cuda/dense_ops.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

#ifdef USE_CUDA
// Test fixture for CUDA dense operations
class CUDADenseOpsTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping CUDA dense ops tests";
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

  bool has_gpu_;
  const Device *gpu_device_;
};

// ==================== compute_dense_forward Tests ====================

TEST_F(CUDADenseOpsTest, DenseForwardBasic) {
  const size_t batch_size = 2;
  const size_t input_features = 3;
  const size_t output_features = 4;

  std::vector<float> input_data(batch_size * input_features);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<float> weight_data(input_features * output_features);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * output_features, 0.0f);
  cpu::dense::compute_dense_forward(input_data.data(), weight_data.data(), cpu_output.data(),
                                    batch_size, input_features, output_features);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * output_features);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_dense_forward_gpu", cuda::dense::compute_dense_forward<float>, gpu_input.get(),
      gpu_weight.get(), gpu_output.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_output_cpu(batch_size * output_features);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * output_features) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

TEST_F(CUDADenseOpsTest, DenseForwardLargeBatch) {
  const size_t batch_size = 32;
  const size_t input_features = 128;
  const size_t output_features = 64;

  std::vector<float> input_data(batch_size * input_features);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100) * 0.01f;
  }

  std::vector<float> weight_data(input_features * output_features);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i % 50) * 0.02f;
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * output_features, 0.0f);
  cpu::dense::compute_dense_forward(input_data.data(), weight_data.data(), cpu_output.data(),
                                    batch_size, input_features, output_features);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * output_features);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_dense_forward_gpu", cuda::dense::compute_dense_forward<float>, gpu_input.get(),
      gpu_weight.get(), gpu_output.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_output_cpu(batch_size * output_features);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * output_features) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

TEST_F(CUDADenseOpsTest, DenseForwardSingleSample) {
  const size_t batch_size = 1;
  const size_t input_features = 10;
  const size_t output_features = 5;

  std::vector<float> input_data(batch_size * input_features);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1) * 0.5f;
  }

  std::vector<float> weight_data(input_features * output_features);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_output(batch_size * output_features, 0.0f);
  cpu::dense::compute_dense_forward(input_data.data(), weight_data.data(), cpu_output.data(),
                                    batch_size, input_features, output_features);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_output =
      make_array_ptr<float[]>(gpu_device_, batch_size * output_features);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_dense_forward_gpu", cuda::dense::compute_dense_forward<float>, gpu_input.get(),
      gpu_weight.get(), gpu_output.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_output_cpu(batch_size * output_features);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * output_features) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

// ==================== compute_weight_gradients Tests ====================

TEST_F(CUDADenseOpsTest, WeightGradientsBasic) {
  const size_t batch_size = 2;
  const size_t input_features = 3;
  const size_t output_features = 4;

  std::vector<float> input_data(batch_size * input_features);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<float> gradient_data(batch_size * output_features);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_weight_grad(input_features * output_features, 0.0f);
  cpu::dense::compute_weight_gradients(input_data.data(), gradient_data.data(),
                                       cpu_weight_grad.data(), batch_size, input_features,
                                       output_features);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight_grad =
      make_array_ptr<float[]>(gpu_device_, input_features * output_features);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(input_features * output_features, 0.0f);
  gpu_device_->copyToDevice(gpu_weight_grad.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_weight_grad_gpu", cuda::dense::compute_weight_gradients<float>, gpu_input.get(),
      gpu_gradient.get(), gpu_weight_grad.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_weight_grad_cpu(input_features * output_features);
  gpu_device_->copyToHost(gpu_weight_grad_cpu.data(), gpu_weight_grad.get(),
                          (input_features * output_features) * sizeof(float));

  compareArrays(cpu_weight_grad, gpu_weight_grad_cpu);
}

TEST_F(CUDADenseOpsTest, WeightGradientsLarge) {
  const size_t batch_size = 16;
  const size_t input_features = 64;
  const size_t output_features = 32;

  std::vector<float> input_data(batch_size * input_features);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100) * 0.01f;
  }

  std::vector<float> gradient_data(batch_size * output_features);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 50) * 0.02f;
  }

  // CPU version
  std::vector<float> cpu_weight_grad(input_features * output_features, 0.0f);
  cpu::dense::compute_weight_gradients(input_data.data(), gradient_data.data(),
                                       cpu_weight_grad.data(), batch_size, input_features,
                                       output_features);

  // GPU version
  device_ptr<float[]> gpu_input = make_array_ptr<float[]>(gpu_device_, input_data.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight_grad =
      make_array_ptr<float[]>(gpu_device_, input_features * output_features);

  gpu_device_->copyToDevice(gpu_input.get(), input_data.data(), input_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(input_features * output_features, 0.0f);
  gpu_device_->copyToDevice(gpu_weight_grad.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_weight_grad_gpu", cuda::dense::compute_weight_gradients<float>, gpu_input.get(),
      gpu_gradient.get(), gpu_weight_grad.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_weight_grad_cpu(input_features * output_features);
  gpu_device_->copyToHost(gpu_weight_grad_cpu.data(), gpu_weight_grad.get(),
                          (input_features * output_features) * sizeof(float));

  compareArrays(cpu_weight_grad, gpu_weight_grad_cpu);
}

// ==================== compute_input_gradients Tests ====================

TEST_F(CUDADenseOpsTest, InputGradientsBasic) {
  const size_t batch_size = 2;
  const size_t input_features = 3;
  const size_t output_features = 4;

  std::vector<float> gradient_data(batch_size * output_features);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  std::vector<float> weight_data(input_features * output_features);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * input_features, 0.0f);
  cpu::dense::compute_input_gradients(gradient_data.data(), weight_data.data(),
                                      cpu_grad_input.data(), batch_size, input_features,
                                      output_features);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * input_features);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * input_features, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_input_grad_gpu", cuda::dense::compute_input_gradients<float>, gpu_gradient.get(),
      gpu_weight.get(), gpu_grad_input.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_grad_input_cpu(batch_size * input_features);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          (batch_size * input_features) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDADenseOpsTest, InputGradientsLarge) {
  const size_t batch_size = 16;
  const size_t input_features = 64;
  const size_t output_features = 32;

  std::vector<float> gradient_data(batch_size * output_features);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 50) * 0.02f;
  }

  std::vector<float> weight_data(input_features * output_features);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<float>(i % 100) * 0.01f;
  }

  // CPU version
  std::vector<float> cpu_grad_input(batch_size * input_features, 0.0f);
  cpu::dense::compute_input_gradients(gradient_data.data(), weight_data.data(),
                                      cpu_grad_input.data(), batch_size, input_features,
                                      output_features);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_weight = make_array_ptr<float[]>(gpu_device_, weight_data.size());
  device_ptr<float[]> gpu_grad_input =
      make_array_ptr<float[]>(gpu_device_, batch_size * input_features);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_weight.get(), weight_data.data(),
                            weight_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * input_features, 0.0f);
  gpu_device_->copyToDevice(gpu_grad_input.get(), zero_grad.data(),
                            zero_grad.size() * sizeof(float));

  auto gpu_task = create_gpu_task(
      "test_input_grad_gpu", cuda::dense::compute_input_gradients<float>, gpu_gradient.get(),
      gpu_weight.get(), gpu_grad_input.get(), batch_size, input_features, output_features);
  gpu_task->sync();

  std::vector<float> gpu_grad_input_cpu(batch_size * input_features);
  gpu_device_->copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get(),
                          (batch_size * input_features) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

// ==================== compute_bias_gradients Tests ====================

TEST_F(CUDADenseOpsTest, BiasGradientsBasic) {
  const size_t batch_size = 2;
  const size_t output_features = 4;

  std::vector<float> gradient_data(batch_size * output_features);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  // CPU version
  std::vector<float> cpu_bias_grad(output_features, 0.0f);
  cpu::dense::compute_bias_gradients(gradient_data.data(), cpu_bias_grad.data(), batch_size,
                                     output_features);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_bias_grad = make_array_ptr<float[]>(gpu_device_, output_features);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_bias_grad(output_features, 0.0f);
  gpu_device_->copyToDevice(gpu_bias_grad.get(), zero_bias_grad.data(),
                            zero_bias_grad.size() * sizeof(float));

  auto gpu_task =
      create_gpu_task("test_bias_grad_gpu", cuda::dense::compute_bias_gradients<float>,
                      gpu_gradient.get(), gpu_bias_grad.get(), batch_size, output_features);
  gpu_task->sync();

  std::vector<float> gpu_bias_grad_cpu(output_features);
  gpu_device_->copyToHost(gpu_bias_grad_cpu.data(), gpu_bias_grad.get(),
                          output_features * sizeof(float));

  compareArrays(cpu_bias_grad, gpu_bias_grad_cpu);
}

TEST_F(CUDADenseOpsTest, BiasGradientsLargeBatch) {
  const size_t batch_size = 32;
  const size_t output_features = 128;

  std::vector<float> gradient_data(batch_size * output_features);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i % 100) * 0.01f;
  }

  // CPU version
  std::vector<float> cpu_bias_grad(output_features, 0.0f);
  cpu::dense::compute_bias_gradients(gradient_data.data(), cpu_bias_grad.data(), batch_size,
                                     output_features);

  // GPU version
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, gradient_data.size());
  device_ptr<float[]> gpu_bias_grad = make_array_ptr<float[]>(gpu_device_, output_features);

  gpu_device_->copyToDevice(gpu_gradient.get(), gradient_data.data(),
                            gradient_data.size() * sizeof(float));

  std::vector<float> zero_bias_grad(output_features, 0.0f);
  gpu_device_->copyToDevice(gpu_bias_grad.get(), zero_bias_grad.data(),
                            zero_bias_grad.size() * sizeof(float));

  auto gpu_task =
      create_gpu_task("test_bias_grad_gpu", cuda::dense::compute_bias_gradients<float>,
                      gpu_gradient.get(), gpu_bias_grad.get(), batch_size, output_features);
  gpu_task->sync();

  std::vector<float> gpu_bias_grad_cpu(output_features);
  gpu_device_->copyToHost(gpu_bias_grad_cpu.data(), gpu_bias_grad.get(),
                          output_features * sizeof(float));

  compareArrays(cpu_bias_grad, gpu_bias_grad_cpu);
}

// ==================== add_bias_vector Tests ====================

TEST_F(CUDADenseOpsTest, AddBiasBasic) {
  const size_t batch_size = 2;
  const size_t output_features = 4;

  std::vector<float> output_data(batch_size * output_features);
  for (size_t i = 0; i < output_data.size(); ++i) {
    output_data[i] = static_cast<float>(i + 1);
  }

  std::vector<float> bias_data(output_features);
  for (size_t i = 0; i < bias_data.size(); ++i) {
    bias_data[i] = static_cast<float>(i + 1) * 0.5f;
  }

  // CPU version
  std::vector<float> cpu_output = output_data;
  cpu::dense::add_bias_vector(cpu_output.data(), bias_data.data(), batch_size, output_features);

  // GPU version
  device_ptr<float[]> gpu_output = make_array_ptr<float[]>(gpu_device_, output_data.size());
  device_ptr<float[]> gpu_bias = make_array_ptr<float[]>(gpu_device_, bias_data.size());

  gpu_device_->copyToDevice(gpu_output.get(), output_data.data(),
                            output_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_bias.get(), bias_data.data(), bias_data.size() * sizeof(float));

  auto gpu_task = create_gpu_task("test_add_bias_gpu", cuda::dense::add_bias_vector<float>,
                                  gpu_output.get(), gpu_bias.get(), batch_size, output_features);
  gpu_task->sync();

  std::vector<float> gpu_output_cpu(batch_size * output_features);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * output_features) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

TEST_F(CUDADenseOpsTest, AddBiasLarge) {
  const size_t batch_size = 32;
  const size_t output_features = 128;

  std::vector<float> output_data(batch_size * output_features);
  for (size_t i = 0; i < output_data.size(); ++i) {
    output_data[i] = static_cast<float>(i % 100) * 0.01f;
  }

  std::vector<float> bias_data(output_features);
  for (size_t i = 0; i < bias_data.size(); ++i) {
    bias_data[i] = static_cast<float>(i % 50) * 0.02f;
  }

  // CPU version
  std::vector<float> cpu_output = output_data;
  cpu::dense::add_bias_vector(cpu_output.data(), bias_data.data(), batch_size, output_features);

  // GPU version
  device_ptr<float[]> gpu_output = make_array_ptr<float[]>(gpu_device_, output_data.size());
  device_ptr<float[]> gpu_bias = make_array_ptr<float[]>(gpu_device_, bias_data.size());

  gpu_device_->copyToDevice(gpu_output.get(), output_data.data(),
                            output_data.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_bias.get(), bias_data.data(), bias_data.size() * sizeof(float));

  auto gpu_task = create_gpu_task("test_add_bias_gpu", cuda::dense::add_bias_vector<float>,
                                  gpu_output.get(), gpu_bias.get(), batch_size, output_features);
  gpu_task->sync();

  std::vector<float> gpu_output_cpu(batch_size * output_features);
  gpu_device_->copyToHost(gpu_output_cpu.data(), gpu_output.get(),
                          (batch_size * output_features) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // USE_CUDA
