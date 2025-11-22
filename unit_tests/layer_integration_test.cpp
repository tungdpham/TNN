/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

#ifdef USE_CUDA

/**
 * Integration test fixture for testing complete layer implementations
 * comparing CPU vs GPU device execution.
 */
class LayerIntegrationTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Initialize devices once for all tests in this suite
    initializeDefaultDevices();
  }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    // Find CPU and GPU devices
    has_cpu_ = false;
    has_gpu_ = false;

    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.getDeviceType() == DeviceType::CPU) {
        cpu_device_ = &device;
        has_cpu_ = true;
      } else if (device.getDeviceType() == DeviceType::GPU) {
        gpu_device_ = &device;
        has_gpu_ = true;
      }
    }

    if (!has_cpu_) {
      GTEST_SKIP() << "No CPU device available";
    }
    if (!has_gpu_) {
      GTEST_SKIP() << "No GPU device available, skipping layer integration tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  // Helper function to compare tensors with tolerance
  void compareTensors(const Tensor<float> &expected, const Tensor<float> &actual,
                      float tolerance = 1e-3f, const std::string &context = "") {
    ASSERT_EQ(expected.shape(), actual.shape()) << context << " Tensors have different shapes";

    // Move both to CPU for comparison
    Tensor<float> expected_cpu = expected.device()->getDeviceType() == DeviceType::CPU
                                     ? expected
                                     : expected.to_device(&getCPU());
    Tensor<float> actual_cpu =
        actual.device()->getDeviceType() == DeviceType::CPU ? actual : actual.to_device(&getCPU());

    size_t total_elements = expected_cpu.size();
    const float *expected_data = expected_cpu.data();
    const float *actual_data = actual_cpu.data();

    size_t mismatch_count = 0;
    const size_t max_mismatches_to_show = 10;

    for (size_t i = 0; i < total_elements; ++i) {
      if (std::abs(expected_data[i] - actual_data[i]) > tolerance) {
        if (mismatch_count < max_mismatches_to_show) {
          std::cerr << context << " Mismatch at index " << i << ": Expected " << expected_data[i]
                    << ", Got " << actual_data[i]
                    << ", Diff: " << std::abs(expected_data[i] - actual_data[i]) << std::endl;
        }
        mismatch_count++;
      }
    }

    EXPECT_EQ(0, mismatch_count) << context << " Found " << mismatch_count << " mismatches out of "
                                 << total_elements << " elements";
  }

  bool has_cpu_;
  bool has_gpu_;
  const Device *cpu_device_;
  const Device *gpu_device_;
};

// ==================== Conv2DLayer Integration Tests ====================

TEST_F(LayerIntegrationTest, Conv2DLayerForwardBasic) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 48;
  const size_t input_h = 28;
  const size_t input_w = 28;
  const size_t kernel_h = 3;
  const size_t kernel_w = 3;
  const size_t stride_h = 1;
  const size_t stride_w = 1;
  const size_t pad_h = 1;
  const size_t pad_w = 1;

  // Create layers on CPU and GPU
  Conv2DLayer<float> cpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, true, "cpu_conv");
  Conv2DLayer<float> gpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, true, "gpu_conv");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Copy weights from CPU to GPU to ensure identical parameters
  *gpu_layer.parameters()[0] = cpu_layer.parameters()[0]->to_device(gpu_device_);
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = cpu_layer.parameters()[1]->to_device(gpu_device_);
  }

  // Create input tensor
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(2.0f); // Range [-1, 1]

  // Forward pass on CPU
  Tensor<float> cpu_output = cpu_layer.forward(input);

  // Forward pass on GPU (input will be copied to GPU inside layer)
  Tensor<float> gpu_output = gpu_layer.forward(input);

  // Compare outputs
  compareTensors(cpu_output, gpu_output, 1e-3f, "Conv2DLayer Forward:");
}

TEST_F(LayerIntegrationTest, Conv2DLayerBackwardBasic) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 4;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t kernel_h = 3;
  const size_t kernel_w = 3;
  const size_t stride_h = 1;
  const size_t stride_w = 1;
  const size_t pad_h = 1;
  const size_t pad_w = 1;

  // Create layers on CPU and GPU
  Conv2DLayer<float> cpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, true, "cpu_conv");
  Conv2DLayer<float> gpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, true, "gpu_conv");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Copy weights from CPU to GPU
  *gpu_layer.parameters()[0] = cpu_layer.parameters()[0]->to_device(gpu_device_);
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = cpu_layer.parameters()[1]->to_device(gpu_device_);
  }

  // Create input tensor and run forward pass
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(2.0f);

  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  // Create gradient tensor for backward pass
  Tensor<float> grad_output({batch_size, out_channels, input_h, input_w}, cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  // Backward pass on CPU
  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);

  // Backward pass on GPU
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  // Compare input gradients
  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "Conv2DLayer Backward Input Gradient:");

  // Compare weight gradients
  compareTensors(*cpu_layer.gradients()[0], *gpu_layer.gradients()[0], 1e-2f,
                 "Conv2DLayer Backward Weight Gradient:");

  // Compare bias gradients if they exist
  if (cpu_layer.gradients().size() > 1) {
    compareTensors(*cpu_layer.gradients()[1], *gpu_layer.gradients()[1], 1e-2f,
                   "Conv2DLayer Backward Bias Gradient:");
  }
}

TEST_F(LayerIntegrationTest, Conv2DLayerStridedConvolution) {
  const size_t batch_size = 1;
  const size_t in_channels = 2;
  const size_t out_channels = 3;
  const size_t input_h = 16;
  const size_t input_w = 16;
  const size_t kernel_h = 5;
  const size_t kernel_w = 5;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t pad_h = 2;
  const size_t pad_w = 2;

  // Create layers
  Conv2DLayer<float> cpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, false, "cpu_conv_strided");
  Conv2DLayer<float> gpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, false, "gpu_conv_strided");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Sync weights
  *gpu_layer.parameters()[0] = cpu_layer.parameters()[0]->to_device(gpu_device_);

  // Test forward and backward
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(1.0f);

  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  compareTensors(cpu_output, gpu_output, 1e-3f, "Conv2DLayer Strided Forward:");

  // Backward pass
  Tensor<float> grad_output(cpu_output.shape(), cpu_device_);
  grad_output.fill_random_uniform(1.0f);

  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "Conv2DLayer Strided Backward:");
  compareTensors(*cpu_layer.gradients()[0], *gpu_layer.gradients()[0], 1e-2f,
                 "Conv2DLayer Strided Weight Gradient:");
}

// ==================== DenseLayer Integration Tests ====================

TEST_F(LayerIntegrationTest, DenseLayerForwardBasic) {
  const size_t batch_size = 4;
  const size_t input_features = 128;
  const size_t output_features = 64;

  // Create layers on CPU and GPU
  DenseLayer<float> cpu_layer(input_features, output_features, true, "cpu_dense");
  DenseLayer<float> gpu_layer(input_features, output_features, true, "gpu_dense");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Sync weights and biases
  *gpu_layer.parameters()[0] = cpu_layer.parameters()[0]->to_device(gpu_device_);
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = cpu_layer.parameters()[1]->to_device(gpu_device_);
  }

  // Create input tensor (Dense layer expects 4D input with H=W=1)
  Tensor<float> input({batch_size, input_features, 1, 1}, cpu_device_);
  input.fill_random_uniform(2.0f);

  // Forward pass
  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  compareTensors(cpu_output, gpu_output, 1e-3f, "DenseLayer Forward:");
}

TEST_F(LayerIntegrationTest, DenseLayerBackwardBasic) {
  const size_t batch_size = 4;
  const size_t input_features = 128;
  const size_t output_features = 64;

  // Create layers
  DenseLayer<float> cpu_layer(input_features, output_features, true, "cpu_dense");
  DenseLayer<float> gpu_layer(input_features, output_features, true, "gpu_dense");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Sync weights
  *gpu_layer.parameters()[0] = cpu_layer.parameters()[0]->to_device(gpu_device_);
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = cpu_layer.parameters()[1]->to_device(gpu_device_);
  }

  // Forward pass
  Tensor<float> input({batch_size, input_features, 1, 1}, cpu_device_);
  input.fill_random_uniform(2.0f);

  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  // Backward pass
  Tensor<float> grad_output({batch_size, output_features, 1, 1}, cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  // Compare gradients
  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "DenseLayer Backward Input Gradient:");
  compareTensors(*cpu_layer.gradients()[0], *gpu_layer.gradients()[0], 1e-2f,
                 "DenseLayer Backward Weight Gradient:");

  if (cpu_layer.gradients().size() > 1) {
    compareTensors(*cpu_layer.gradients()[1], *gpu_layer.gradients()[1], 1e-2f,
                   "DenseLayer Backward Bias Gradient:");
  }
}

TEST_F(LayerIntegrationTest, DenseLayerLargeMatrix) {
  const size_t batch_size = 8;
  const size_t input_features = 512;
  const size_t output_features = 256;

  // Create layers without bias
  DenseLayer<float> cpu_layer(input_features, output_features, false, "cpu_dense_large");
  DenseLayer<float> gpu_layer(input_features, output_features, false, "gpu_dense_large");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Sync weights
  *gpu_layer.parameters()[0] = cpu_layer.parameters()[0]->to_device(gpu_device_);

  // Test
  Tensor<float> input({batch_size, input_features, 1, 1}, cpu_device_);
  input.fill_random_uniform(1.0f);

  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  compareTensors(cpu_output, gpu_output, 1e-3f, "DenseLayer Large Forward:");

  // Backward
  Tensor<float> grad_output({batch_size, output_features, 1, 1}, cpu_device_);
  grad_output.fill_random_uniform(1.0f);

  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "DenseLayer Large Backward:");
}

// ==================== MaxPool2DLayer Integration Tests ====================

TEST_F(LayerIntegrationTest, MaxPool2DLayerForwardBasic) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t pad_h = 0;
  const size_t pad_w = 0;

  // Create layers
  MaxPool2DLayer<float> cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool");
  MaxPool2DLayer<float> gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Create input
  Tensor<float> input({batch_size, channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(10.0f);

  // Forward pass
  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  compareTensors(cpu_output, gpu_output, 1e-4f, "MaxPool2DLayer Forward:");
}

TEST_F(LayerIntegrationTest, MaxPool2DLayerBackwardBasic) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t pad_h = 0;
  const size_t pad_w = 0;

  // Create layers
  MaxPool2DLayer<float> cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool");
  MaxPool2DLayer<float> gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Create input
  Tensor<float> input({batch_size, channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(20.0f);

  // Forward pass
  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  // Backward pass
  Tensor<float> grad_output(cpu_output.shape(), cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f, "MaxPool2DLayer Backward:");
}

TEST_F(LayerIntegrationTest, MaxPool2DLayerWithPadding) {
  const size_t batch_size = 1;
  const size_t channels = 4;
  const size_t input_h = 7;
  const size_t input_w = 7;
  const size_t pool_h = 3;
  const size_t pool_w = 3;
  const size_t stride_h = 1;
  const size_t stride_w = 1;
  const size_t pad_h = 1;
  const size_t pad_w = 1;

  // Create layers
  MaxPool2DLayer<float> cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                  "cpu_maxpool_pad");
  MaxPool2DLayer<float> gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                  "gpu_maxpool_pad");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Create input
  Tensor<float> input({batch_size, channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(10.0f);

  // Forward pass
  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  compareTensors(cpu_output, gpu_output, 1e-4f, "MaxPool2DLayer Padded Forward:");

  // Backward pass
  Tensor<float> grad_output(cpu_output.shape(), cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f, "MaxPool2DLayer Padded Backward:");
}

TEST_F(LayerIntegrationTest, MaxPool2DLayerNonSquare) {
  const size_t batch_size = 3;
  const size_t channels = 2;
  const size_t input_h = 12;
  const size_t input_w = 16;
  const size_t pool_h = 3;
  const size_t pool_w = 4;
  const size_t stride_h = 3;
  const size_t stride_w = 4;
  const size_t pad_h = 0;
  const size_t pad_w = 0;

  // Create layers with non-square pooling
  MaxPool2DLayer<float> cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                  "cpu_maxpool_nonsq");
  MaxPool2DLayer<float> gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                  "gpu_maxpool_nonsq");

  cpu_layer.set_device(cpu_device_);
  gpu_layer.set_device(gpu_device_);

  cpu_layer.initialize();
  gpu_layer.initialize();

  // Create input
  Tensor<float> input({batch_size, channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(16.0f);

  // Forward pass
  Tensor<float> cpu_output = cpu_layer.forward(input);
  Tensor<float> gpu_output = gpu_layer.forward(input);

  compareTensors(cpu_output, gpu_output, 1e-4f, "MaxPool2DLayer Non-square Forward:");

  // Backward pass
  Tensor<float> grad_output(cpu_output.shape(), cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  Tensor<float> cpu_grad_input = cpu_layer.backward(grad_output);
  Tensor<float> gpu_grad_input = gpu_layer.backward(grad_output);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f, "MaxPool2DLayer Non-square Backward:");
}

// ==================== Multi-Layer Pipeline Tests ====================

TEST_F(LayerIntegrationTest, Conv2DMaxPoolPipeline) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 16;
  const size_t input_w = 16;

  // CPU pipeline
  Conv2DLayer<float> cpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, true, "cpu_conv");
  MaxPool2DLayer<float> cpu_pool(2, 2, 2, 2, 0, 0, "cpu_pool");

  cpu_conv.set_device(cpu_device_);
  cpu_pool.set_device(cpu_device_);
  cpu_conv.initialize();
  cpu_pool.initialize();

  // GPU pipeline
  Conv2DLayer<float> gpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, true, "gpu_conv");
  MaxPool2DLayer<float> gpu_pool(2, 2, 2, 2, 0, 0, "gpu_pool");

  gpu_conv.set_device(gpu_device_);
  gpu_pool.set_device(gpu_device_);
  gpu_conv.initialize();
  gpu_pool.initialize();

  // Sync conv weights
  *gpu_conv.parameters()[0] = cpu_conv.parameters()[0]->to_device(gpu_device_);
  *gpu_conv.parameters()[1] = cpu_conv.parameters()[1]->to_device(gpu_device_);

  // Input
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(2.0f);

  // Forward pass - CPU
  Tensor<float> cpu_conv_out = cpu_conv.forward(input);
  Tensor<float> cpu_pool_out = cpu_pool.forward(cpu_conv_out);

  // Forward pass - GPU
  Tensor<float> gpu_conv_out = gpu_conv.forward(input);
  Tensor<float> gpu_pool_out = gpu_pool.forward(gpu_conv_out);

  compareTensors(cpu_pool_out, gpu_pool_out, 1e-3f, "Conv2D-MaxPool Pipeline Forward:");

  // Backward pass
  Tensor<float> grad_output(cpu_pool_out.shape(), cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  // CPU backward
  Tensor<float> cpu_grad_pool = cpu_pool.backward(grad_output);
  Tensor<float> cpu_grad_conv = cpu_conv.backward(cpu_grad_pool);

  // GPU backward
  Tensor<float> gpu_grad_pool = gpu_pool.backward(grad_output);
  Tensor<float> gpu_grad_conv = gpu_conv.backward(gpu_grad_pool);

  compareTensors(cpu_grad_conv, gpu_grad_conv, 1e-2f, "Conv2D-MaxPool Pipeline Backward:");
}

TEST_F(LayerIntegrationTest, Conv2DDensePipeline) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t dense_output = 10;

  // Calculate flattened size after conv: (8x8 output) * 8 channels = 512
  const size_t flattened_size = input_h * input_w * out_channels;

  // CPU pipeline
  Conv2DLayer<float> cpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, false, "cpu_conv");
  DenseLayer<float> cpu_dense(flattened_size, dense_output, true, "cpu_dense");

  cpu_conv.set_device(cpu_device_);
  cpu_dense.set_device(cpu_device_);
  cpu_conv.initialize();
  cpu_dense.initialize();

  // GPU pipeline
  Conv2DLayer<float> gpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, false, "gpu_conv");
  DenseLayer<float> gpu_dense(flattened_size, dense_output, true, "gpu_dense");

  gpu_conv.set_device(gpu_device_);
  gpu_dense.set_device(gpu_device_);
  gpu_conv.initialize();
  gpu_dense.initialize();

  // Sync weights
  *gpu_conv.parameters()[0] = cpu_conv.parameters()[0]->to_device(gpu_device_);
  *gpu_dense.parameters()[0] = cpu_dense.parameters()[0]->to_device(gpu_device_);
  *gpu_dense.parameters()[1] = cpu_dense.parameters()[1]->to_device(gpu_device_);

  // Input
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, cpu_device_);
  input.fill_random_uniform(2.0f);

  // Forward - CPU
  Tensor<float> cpu_conv_out = cpu_conv.forward(input);
  // Reshape for dense layer
  Tensor<float> cpu_conv_flat = cpu_conv_out.reshape({batch_size, flattened_size, 1, 1});
  Tensor<float> cpu_dense_out = cpu_dense.forward(cpu_conv_flat);

  // Forward - GPU
  Tensor<float> gpu_conv_out = gpu_conv.forward(input);
  Tensor<float> gpu_conv_flat = gpu_conv_out.reshape({batch_size, flattened_size, 1, 1});
  Tensor<float> gpu_dense_out = gpu_dense.forward(gpu_conv_flat);

  compareTensors(cpu_dense_out, gpu_dense_out, 1e-3f, "Conv2D-Dense Pipeline Forward:");

  // Backward
  Tensor<float> grad_output({batch_size, dense_output, 1, 1}, cpu_device_);
  grad_output.fill_random_uniform(2.0f);

  // CPU backward
  Tensor<float> cpu_grad_dense = cpu_dense.backward(grad_output);
  Tensor<float> cpu_grad_dense_reshape = cpu_grad_dense.reshape(cpu_conv_out.shape());
  Tensor<float> cpu_grad_conv = cpu_conv.backward(cpu_grad_dense_reshape);

  // GPU backward
  Tensor<float> gpu_grad_dense = gpu_dense.backward(grad_output);
  Tensor<float> gpu_grad_dense_reshape = gpu_grad_dense.reshape(gpu_conv_out.shape());
  Tensor<float> gpu_grad_conv = gpu_conv.backward(gpu_grad_dense_reshape);

  compareTensors(cpu_grad_conv, gpu_grad_conv, 1e-2f, "Conv2D-Dense Pipeline Backward:");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // USE_CUDA
