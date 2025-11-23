/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

/**
 * Test fixture for SequentialBuilder residual block tests.
 * These tests verify that the SequentialBuilder correctly constructs
 * basic and bottleneck residual blocks with proper dimensions and forward/backward passes.
 */
class SequentialResidualBlockTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    has_cpu_ = false;
    cpu_device_ = nullptr;

    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.getDeviceType() == DeviceType::CPU) {
        cpu_device_ = &device;
        has_cpu_ = true;
        break;
      }
    }

    if (!has_cpu_) {
      GTEST_SKIP() << "No CPU device available";
    }
  }

  /**
   * Helper to verify output shape matches expected shape
   */
  void verify_output_shape(const std::vector<size_t> &actual, const std::vector<size_t> &expected,
                           const std::string &test_name = "") {
    ASSERT_EQ(actual.size(), expected.size()) << test_name << ": Shape dimension mismatch";
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(actual[i], expected[i])
          << test_name << ": Dimension " << i << " mismatch. Expected: " << expected[i]
          << ", Got: " << actual[i];
    }
  }

  /**
   * Helper to verify numerical output values
   */
  void verify_output_values(const Tensor<float> &output, float expected_min, float expected_max,
                            const std::string &test_name = "") {
    const float *output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_GE(output_data[i], expected_min)
          << test_name << ": Value at index " << i << " below minimum";
      EXPECT_LE(output_data[i], expected_max)
          << test_name << ": Value at index " << i << " above maximum";
    }
  }

  bool has_cpu_;
  const Device *cpu_device_;
};

// ============================================================================
// Basic Residual Block Tests
// ============================================================================

TEST_F(SequentialResidualBlockTest, BasicResidualBlockIdentityShortcut) {
  // Test case: same in_channels and stride=1 (identity shortcut)
  Sequential<float> model = SequentialBuilder<float>("test_basic_identity")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 1, "basic_64_64")
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output shape should match input shape
  verify_output_shape(output.shape(), {1, 64, 32, 32}, "BasicResidualBlockIdentityShortcut");

  // Output should be non-zero after forward pass
  const float *output_data = output.data();
  bool has_nonzero = false;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockProjectionShortcut) {
  // Test case: different out_channels (requires projection shortcut)
  Sequential<float> model = SequentialBuilder<float>("test_basic_projection")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 128, 1, "basic_64_128")
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output channels should be 128
  verify_output_shape(output.shape(), {1, 128, 32, 32}, "BasicResidualBlockProjectionShortcut");

  // Output should be non-zero
  const float *output_data = output.data();
  bool has_nonzero = false;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockStridedShortcut) {
  // Test case: stride=2 (requires projection shortcut)
  Sequential<float> model = SequentialBuilder<float>("test_basic_strided")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 2, "basic_64_64_stride2")
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output spatial dimensions should be halved due to stride=2
  verify_output_shape(output.shape(), {1, 64, 16, 16}, "BasicResidualBlockStridedShortcut");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockStridedAndProjection) {
  // Test case: stride=2 and channel change (both require projection)
  Sequential<float> model = SequentialBuilder<float>("test_basic_strided_projection")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 128, 2, "basic_64_128_stride2")
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output should have new channels and halved spatial dimensions
  verify_output_shape(output.shape(), {1, 128, 16, 16}, "BasicResidualBlockStridedAndProjection");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockBackward) {
  // Test backward pass through basic residual block
  Sequential<float> model = SequentialBuilder<float>("test_basic_backward")
                                .input({32, 16, 16})
                                .basic_residual_block(32, 32, 1, "basic_32_32")
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 32, 16, 16}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);
  EXPECT_EQ(output.shape(), input.shape());

  // Create gradient tensor
  Tensor<float> grad_output({1, 32, 16, 16}, cpu_device_);
  grad_output.fill(1.0f);

  auto grad_input = model.backward(grad_output);

  // Gradient input should have same shape as input
  verify_output_shape(grad_input.shape(), input.shape(), "BasicResidualBlockBackward");

  // Gradient should be non-zero
  const float *grad_data = grad_input.data();
  bool has_nonzero = false;
  for (size_t i = 0; i < grad_input.size(); ++i) {
    if (grad_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockMultipleBlocks) {
  // Test multiple stacked basic residual blocks (like ResNet-18/34)
  Sequential<float> model = SequentialBuilder<float>("test_multi_basic")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 1)
                                .basic_residual_block(64, 64, 1)
                                .basic_residual_block(64, 128, 2)
                                .basic_residual_block(128, 128, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Final output should have 128 channels and halved spatial dims
  verify_output_shape(output.shape(), {1, 128, 16, 16}, "BasicResidualBlockMultipleBlocks");
}

// ============================================================================
// Bottleneck Residual Block Tests
// ============================================================================

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockIdentityShortcut) {
  // Test case: same in_channels and stride=1 (identity shortcut)
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_identity")
                                .input({256, 32, 32})
                                .bottleneck_residual_block(256, 64, 256, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 256, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output shape should match input shape
  verify_output_shape(output.shape(), {1, 256, 32, 32}, "BottleneckResidualBlockIdentityShortcut");

  // Output should be non-zero
  const float *output_data = output.data();
  bool has_nonzero = false;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockProjectionShortcut) {
  // Test case: different out_channels (requires projection shortcut)
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_projection")
                                .input({64, 32, 32})
                                .bottleneck_residual_block(64, 64, 256, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output channels should be 256 (out_channels)
  verify_output_shape(output.shape(), {1, 256, 32, 32},
                      "BottleneckResidualBlockProjectionShortcut");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockStridedShortcut) {
  // Test case: stride=2 (requires projection shortcut)
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_strided")
                                .input({256, 32, 32})
                                .bottleneck_residual_block(256, 64, 256, 2)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 256, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output spatial dimensions should be halved due to stride=2
  verify_output_shape(output.shape(), {1, 256, 16, 16}, "BottleneckResidualBlockStridedShortcut");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockStridedAndProjection) {
  // Test case: stride=2 and channel change (both require projection)
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_strided_projection")
                                .input({64, 32, 32})
                                .bottleneck_residual_block(64, 64, 256, 2)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Output should have 256 channels and halved spatial dimensions
  verify_output_shape(output.shape(), {1, 256, 16, 16},
                      "BottleneckResidualBlockStridedAndProjection");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockBackward) {
  // Test backward pass through bottleneck residual block
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_backward")
                                .input({64, 16, 16})
                                .bottleneck_residual_block(64, 32, 64, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 16, 16}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);
  EXPECT_EQ(output.shape(), input.shape());

  // Create gradient tensor
  Tensor<float> grad_output({1, 64, 16, 16}, cpu_device_);
  grad_output.fill(1.0f);

  auto grad_input = model.backward(grad_output);

  // Gradient input should have same shape as input
  verify_output_shape(grad_input.shape(), input.shape(), "BottleneckResidualBlockBackward");

  // Gradient should be non-zero
  const float *grad_data = grad_input.data();
  bool has_nonzero = false;
  for (size_t i = 0; i < grad_input.size(); ++i) {
    if (grad_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockMultipleBlocks) {
  // Test multiple stacked bottleneck residual blocks (like ResNet-50/101/152)
  Sequential<float> model = SequentialBuilder<float>("test_multi_bottleneck")
                                .input({64, 32, 32})
                                .bottleneck_residual_block(64, 64, 256, 1)
                                .bottleneck_residual_block(256, 64, 256, 1)
                                .bottleneck_residual_block(256, 128, 512, 2)
                                .bottleneck_residual_block(512, 128, 512, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Final output should have 512 channels and halved spatial dims
  verify_output_shape(output.shape(), {1, 512, 16, 16}, "BottleneckResidualBlockMultipleBlocks");
}

// ============================================================================
// Mixed Basic and Bottleneck Tests
// ============================================================================

TEST_F(SequentialResidualBlockTest, MixedBasicAndBottleneckBlocks) {
  // Test a model mixing basic and bottleneck residual blocks
  Sequential<float> model = SequentialBuilder<float>("test_mixed_residuals")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 1)
                                .bottleneck_residual_block(64, 64, 256, 1)
                                .basic_residual_block(256, 256, 1)
                                .bottleneck_residual_block(256, 128, 512, 2)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Final output should have 512 channels and halved spatial dims
  verify_output_shape(output.shape(), {1, 512, 16, 16}, "MixedBasicAndBottleneckBlocks");
}

TEST_F(SequentialResidualBlockTest, ResNet18LikeArchitecture) {
  // Simplified ResNet-18 structure (without initial conv and avgpool)
  Sequential<float> model = SequentialBuilder<float>("resnet18_like")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 1)
                                .basic_residual_block(64, 64, 1)
                                .basic_residual_block(64, 128, 2)
                                .basic_residual_block(128, 128, 1)
                                .basic_residual_block(128, 256, 2)
                                .basic_residual_block(256, 256, 1)
                                .basic_residual_block(256, 512, 2)
                                .basic_residual_block(512, 512, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Final output should be 512 channels and 4x4 spatial dims (32 / 2^3)
  verify_output_shape(output.shape(), {1, 512, 4, 4}, "ResNet18LikeArchitecture");
}

TEST_F(SequentialResidualBlockTest, ResNet50LikeArchitecture) {
  // Simplified ResNet-50 structure (without initial conv and avgpool)
  Sequential<float> model = SequentialBuilder<float>("resnet50_like")
                                .input({64, 32, 32})
                                .bottleneck_residual_block(64, 64, 256, 1)
                                .bottleneck_residual_block(256, 64, 256, 1)
                                .bottleneck_residual_block(256, 64, 256, 1)
                                .bottleneck_residual_block(256, 128, 512, 2)
                                .bottleneck_residual_block(512, 128, 512, 1)
                                .bottleneck_residual_block(512, 128, 512, 1)
                                .bottleneck_residual_block(512, 256, 1024, 2)
                                .bottleneck_residual_block(1024, 256, 1024, 1)
                                .bottleneck_residual_block(1024, 256, 1024, 1)
                                .build();

  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 64, 32, 32}, cpu_device_);
  input.fill(1.0f);

  auto output = model.forward(input);

  // Final output should be 1024 channels and 8x8 spatial dims (32 / 2^2)
  verify_output_shape(output.shape(), {1, 1024, 8, 8}, "ResNet50LikeArchitecture");
}

// ============================================================================
// Output Shape Computation Tests
// ============================================================================

TEST_F(SequentialResidualBlockTest, BasicResidualBlockOutputShapeComputation) {
  // Test that compute_output_shape works correctly for basic residual blocks
  Sequential<float> model = SequentialBuilder<float>("test_basic_shape_comp")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 128, 2)
                                .build();

  std::vector<size_t> input_shape = {1, 64, 32, 32};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 128, 16, 16}, "BasicResidualBlockOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockOutputShapeComputation) {
  // Test that compute_output_shape works correctly for bottleneck residual blocks
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_shape_comp")
                                .input({64, 32, 32})
                                .bottleneck_residual_block(64, 64, 256, 2)
                                .build();

  std::vector<size_t> input_shape = {1, 64, 32, 32};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 256, 16, 16},
                      "BottleneckResidualBlockOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, ChainedOutputShapeComputation) {
  // Test compute_output_shape for chained residual blocks
  Sequential<float> model = SequentialBuilder<float>("test_chained_shape_comp")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 1)
                                .basic_residual_block(64, 128, 2)
                                .bottleneck_residual_block(128, 64, 256, 1)
                                .build();

  std::vector<size_t> input_shape = {1, 64, 32, 32};
  auto output_shape = model.compute_output_shape(input_shape);

  // Should be: 64->64->128 (halved to 16x16) -> 256 (same spatial dims)
  verify_output_shape(output_shape, {1, 256, 16, 16}, "ChainedOutputShapeComputation");
}

// ============================================================================
// Configuration and Serialization Tests
// ============================================================================

TEST_F(SequentialResidualBlockTest, BasicResidualBlockGetConfig) {
  // Test that model configuration can be retrieved
  Sequential<float> model = SequentialBuilder<float>("test_basic_config")
                                .input({64, 32, 32})
                                .basic_residual_block(64, 64, 1, "my_basic_block")
                                .build();

  auto config = model.get_config();

  EXPECT_EQ(config["name"], "test_basic_config");
  EXPECT_TRUE(config.contains("layers"));
  EXPECT_GT(config["layers"].size(), 0);
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockGetConfig) {
  // Test that bottleneck model configuration can be retrieved
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_config")
                                .input({64, 32, 32})
                                .bottleneck_residual_block(64, 64, 256, 1, "my_bottleneck_block")
                                .build();

  auto config = model.get_config();

  EXPECT_EQ(config["name"], "test_bottleneck_config");
  EXPECT_TRUE(config.contains("layers"));
  EXPECT_GT(config["layers"].size(), 0);
}

// ============================================================================
// Numerical Stability and Gradient Tests
// ============================================================================

TEST_F(SequentialResidualBlockTest, BasicResidualBlockNumericalStability) {
  // Test numerical stability with various input ranges
  Sequential<float> model = SequentialBuilder<float>("test_basic_stability")
                                .input({16, 8, 8})
                                .basic_residual_block(16, 16, 1)
                                .build();
  model.set_device(cpu_device_);
  model.initialize();

  // Test with different input scales
  std::vector<float> scales = {0.01f, 0.1f, 1.0f, 10.0f};
  for (float scale : scales) {
    Tensor<float> input({1, 16, 8, 8}, cpu_device_);
    float *input_data = input.data();
    srand(42);
    for (size_t i = 0; i < input.size(); ++i) {
      input_data[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }

    auto output = model.forward(input);

    // Check that output contains finite values (no NaN or Inf)
    const float *output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_TRUE(std::isfinite(output_data[i]))
          << "Output contains non-finite value at index " << i << " with scale " << scale;
    }

    // Check output statistics
    float output_sum = 0.0f;
    float output_abs_max = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
      output_sum += output_data[i];
      output_abs_max = std::max(output_abs_max, std::abs(output_data[i]));
    }
    EXPECT_GT(output_abs_max, 0.0f) << "Output should have non-zero values with scale " << scale;
  }
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockNumericalStability) {
  // Test numerical stability with various input ranges
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_stability")
                                .input({32, 8, 8})
                                .bottleneck_residual_block(32, 16, 32, 1)
                                .build();
  model.set_device(cpu_device_);
  model.initialize();

  // Test with different input scales
  std::vector<float> scales = {0.01f, 0.1f, 1.0f, 10.0f};
  for (float scale : scales) {
    Tensor<float> input({1, 32, 8, 8}, cpu_device_);
    float *input_data = input.data();
    srand(42);
    for (size_t i = 0; i < input.size(); ++i) {
      input_data[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }

    auto output = model.forward(input);

    // Check that output contains finite values (no NaN or Inf)
    const float *output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_TRUE(std::isfinite(output_data[i]))
          << "Output contains non-finite value at index " << i << " with scale " << scale;
    }

    // Check output statistics
    float output_abs_max = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
      output_abs_max = std::max(output_abs_max, std::abs(output_data[i]));
    }
    EXPECT_GT(output_abs_max, 0.0f) << "Output should have non-zero values with scale " << scale;
  }
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockGradientFiniteness) {
  // Test that gradients are finite (no NaN or Inf) for basic residual block
  Sequential<float> model = SequentialBuilder<float>("test_basic_grad_finite")
                                .input({8, 8, 8})
                                .basic_residual_block(8, 8, 1)
                                .build();
  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 8, 8, 8}, cpu_device_);
  srand(42);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  auto output = model.forward(input);
  Tensor<float> grad_output(output.shape(), cpu_device_);
  grad_output.fill(1.0f);

  auto grad_input = model.backward(grad_output);

  // Check that all gradients are finite
  const float *grad_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_TRUE(std::isfinite(grad_data[i])) << "Gradient contains non-finite value at index " << i;
  }

  // Verify gradients are non-zero somewhere
  bool has_nonzero_grad = false;
  for (size_t i = 0; i < grad_input.size(); ++i) {
    if (std::abs(grad_data[i]) > 1e-6f) {
      has_nonzero_grad = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero_grad) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockGradientFiniteness) {
  // Test that gradients are finite (no NaN or Inf) for bottleneck residual block
  Sequential<float> model = SequentialBuilder<float>("test_bottleneck_grad_finite")
                                .input({16, 8, 8})
                                .bottleneck_residual_block(16, 8, 16, 1)
                                .build();
  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 16, 8, 8}, cpu_device_);
  srand(42);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  auto output = model.forward(input);
  Tensor<float> grad_output(output.shape(), cpu_device_);
  grad_output.fill(1.0f);

  auto grad_input = model.backward(grad_output);

  // Check that all gradients are finite
  const float *grad_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_TRUE(std::isfinite(grad_data[i])) << "Gradient contains non-finite value at index " << i;
  }

  // Verify gradients are non-zero somewhere
  bool has_nonzero_grad = false;
  for (size_t i = 0; i < grad_input.size(); ++i) {
    if (std::abs(grad_data[i]) > 1e-6f) {
      has_nonzero_grad = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero_grad) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, ResidualBlockGradientMagnitudes) {
  // Test that gradient magnitudes are reasonable and not exploding/vanishing
  Sequential<float> model = SequentialBuilder<float>("test_grad_magnitudes")
                                .input({16, 16, 16})
                                .basic_residual_block(16, 16, 1)
                                .basic_residual_block(16, 32, 2)
                                .bottleneck_residual_block(32, 16, 32, 1)
                                .build();
  model.set_device(cpu_device_);
  model.initialize();

  Tensor<float> input({1, 16, 16, 16}, cpu_device_);
  srand(42);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = 0.1f * static_cast<float>(rand()) / RAND_MAX;
  }

  auto output = model.forward(input);
  Tensor<float> grad_output(output.shape(), cpu_device_);
  grad_output.fill(0.01f); // Small gradient signal

  auto grad_input = model.backward(grad_output);

  // Compute gradient statistics
  const float *grad_data = grad_input.data();
  float grad_max = -std::numeric_limits<float>::max();
  size_t nonzero_count = 0;

  for (size_t i = 0; i < grad_input.size(); ++i) {
    float abs_grad = std::abs(grad_data[i]);
    if (abs_grad > 1e-10f) {
      grad_max = std::max(grad_max, abs_grad);
      nonzero_count++;
    }
  }

  // Check that gradients are not exploding (max too large)
  EXPECT_LT(grad_max, 100.0f) << "Gradient values are too large (exploding gradients)";

  // Check that we have a reasonable number of non-zero gradients
  EXPECT_GT(nonzero_count, grad_input.size() * 0.1f)
      << "Too many vanishing gradients: only " << nonzero_count << " non-zero out of "
      << grad_input.size();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
