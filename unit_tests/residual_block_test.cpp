/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/blocks_impl/residual_block.hpp"
#include "nn/layers.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

/**
 * Test fixture for ResidualBlock validation tests.
 * These tests verify the mathematical correctness of residual block operations
 * including forward pass (skip connection addition) and backward pass (gradient distribution).
 */
class ResidualBlockTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    has_cpu_ = false;
    cpu_device_ = {};

    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.device_type() == DeviceType::CPU) {
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
   * Verify forward pass mathematically for identity shortcut
   * output = activation(F(x) + x)
   */
  void verify_identity_shortcut_forward(const Tensor<float> &main_path_output,
                                        const Tensor<float> &actual_output,
                                        const std::string &activation_type = "relu",
                                        float tolerance = 1e-5f) {
    EXPECT_EQ(main_path_output.shape(), actual_output.shape());

    const float *main_data = main_path_output.data();
    const float *output_data = actual_output.data();

    std::vector<float> expected_output(actual_output.size());
    for (size_t i = 0; i < actual_output.size(); ++i) {
      // For identity shortcut, F(x) + x
      expected_output[i] = main_data[i];

      // Apply activation
      if (activation_type == "relu") {
        expected_output[i] = std::max(0.0f, expected_output[i]);
      } else if (activation_type == "none" || activation_type == "linear") {
        // No activation
      }
    }

    for (size_t i = 0; i < actual_output.size(); ++i) {
      EXPECT_NEAR(output_data[i], expected_output[i], tolerance)
          << "Mismatch at index " << i << ". Expected: " << expected_output[i]
          << ", Got: " << output_data[i];
    }
  }

  /**
   * Verify backward pass for residual block
   * Gradients are summed from both paths: grad_input = grad_main + grad_shortcut
   */
  void verify_backward_gradient_distribution(const Tensor<float> &grad_main,
                                             const Tensor<float> &grad_shortcut,
                                             const Tensor<float> &actual_grad_input,
                                             float tolerance = 1e-5f) {
    EXPECT_EQ(grad_main.shape(), actual_grad_input.shape());
    EXPECT_EQ(grad_shortcut.shape(), actual_grad_input.shape());

    const float *grad_main_data = grad_main.data();
    const float *grad_shortcut_data = grad_shortcut.data();
    const float *actual_data = actual_grad_input.data();

    for (size_t i = 0; i < actual_grad_input.size(); ++i) {
      float expected = grad_main_data[i] + grad_shortcut_data[i];
      EXPECT_NEAR(actual_data[i], expected, tolerance)
          << "Gradient mismatch at index " << i << ". Expected: " << expected
          << ", Got: " << actual_data[i];
    }
  }

  /**
   * Create a simple linear layer (y = scale * x) for testing
   */
  std::vector<std::unique_ptr<Layer<float>>> create_scaling_layer(float scale,
                                                                  const std::string &name = "scale",
                                                                  size_t in_channels = 1,
                                                                  size_t out_channels = 1) {
    std::vector<std::unique_ptr<Layer<float>>> layers;
    // Use a 1x1 conv with specific initialization to act as a simple linear transformation
    auto layer = conv2d_layer<float>(in_channels, out_channels, 1, 1, 1, 1, 0, 0, false, name);
    layer->set_device(cpu_device_);
    layer->initialize();

    // Set weights to scale value
    auto params = layer->parameters();
    if (!params.empty()) {
      float *weight_data = params[0]->data();
      for (size_t i = 0; i < params[0]->size(); ++i) {
        weight_data[i] = scale;
      }
    }

    layers.push_back(std::move(layer));
    return layers;
  }

  bool has_cpu_;
  const Device *cpu_device_;
};

// Identity Shortcut Tests

TEST_F(ResidualBlockTest, IdentityShortcutForward) {
  // Create simple main path: single layer that multiplies by 2
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(2.0f, "scale_2x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "identity_residual");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  EXPECT_EQ(output.shape(), input.shape());

  // Expected: F(x) + x = 2*1 + 1 = 3
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 3.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, IdentityShortcutForwardWithReLU) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(-2.0f, "scale_neg2x");

  ResidualBlock<float> residual(std::move(main_path), {}, "relu", "identity_relu");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: relu(F(x) + x) = relu(-2*1 + 1) = relu(-1) = 0
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, IdentityShortcutMultiChannel) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(0.5f, "scale_half", 2, 2);

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "identity_multichannel");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 2, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 8; ++i) {
    input_data[i] = 2.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  EXPECT_EQ(output.batch_size(), 1);
  EXPECT_EQ(output.channels(), 2);
  EXPECT_EQ(output.height(), 2);
  EXPECT_EQ(output.width(), 2);

  // Expected: F(x) + x where F is conv2d with scale 0.5
  // With 2 input channels and 2 output channels: each output = sum(0.5 * input[i]) + input
  // = (0.5 * 2 + 0.5 * 2) + 2 = 2 + 2 = 4
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 4.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, IdentityShortcutMultiBatch) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "identity_multibatch");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({2, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 8; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  EXPECT_EQ(output.batch_size(), 2);
  EXPECT_EQ(output.channels(), 1);

  // Expected: F(x) + x = 1*x + x = 2*x
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    float expected = 2.0f * input_data[i];
    EXPECT_NEAR(output_data[i], expected, 1e-5f);
  }
}

// Projection Shortcut Tests

TEST_F(ResidualBlockTest, ProjectionShortcutForward) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(0.5f, "scale_main");

  // Projection shortcut: 1x1 conv with scale 0.25
  std::vector<std::unique_ptr<Layer<float>>> shortcut =
      create_scaling_layer(0.25f, "scale_shortcut");

  ResidualBlock<float> residual(std::move(main_path), std::move(shortcut), "none",
                                "projection_residual");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 4.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: F(x) + shortcut(x) = 0.5*4 + 0.25*4 = 2 + 1 = 3
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 3.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, ProjectionShortcutWithReLU) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(-1.0f, "scale_neg");

  std::vector<std::unique_ptr<Layer<float>>> shortcut = create_scaling_layer(0.5f, "scale_short");

  ResidualBlock<float> residual(std::move(main_path), std::move(shortcut), "relu",
                                "projection_relu");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 2.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: relu(F(x) + shortcut(x)) = relu(-1*2 + 0.5*2) = relu(-1) = 0
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

// Backward Pass Tests

TEST_F(ResidualBlockTest, IdentityShortcutBackward) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(2.0f, "scale_2x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "identity_backward");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  float *grad_data = gradient.data();
  for (int i = 0; i < 4; ++i) {
    grad_data[i] = 1.0f;
  }

  Tensor<float> grad_input(input.shape(), cpu_device_);
  residual.backward(gradient, grad_input);

  EXPECT_EQ(grad_input.shape(), input.shape());

  // Gradient through linear path + gradient through shortcut
  // Both contribute equally in identity shortcut: grad = grad_main + grad_shortcut
  const float *grad_input_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    // grad_main from scaling layer (2.0) * incoming_gradient (1.0) = 2.0
    // grad_shortcut from identity = 1.0
    // Total: 2.0 + 1.0 = 3.0
    EXPECT_NEAR(grad_input_data[i], 3.0f, 1e-4f);
  }
}

TEST_F(ResidualBlockTest, GetConfig) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale");

  ResidualBlock<float> residual(std::move(main_path), {}, "relu", "test_residual");

  LayerConfig config = residual.get_config();

  EXPECT_EQ(config.name, "test_residual");
  EXPECT_EQ(config.get<std::string>("activation"), "relu");
  EXPECT_EQ(config.get<bool>("has_projection"), false);
}

TEST_F(ResidualBlockTest, GetConfigWithProjection) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_main");

  std::vector<std::unique_ptr<Layer<float>>> shortcut;
  shortcut = create_scaling_layer(0.5f, "scale_short");

  ResidualBlock<float> residual(std::move(main_path), std::move(shortcut), "relu",
                                "test_projection");

  LayerConfig config = residual.get_config();

  EXPECT_EQ(config.name, "test_projection");
  EXPECT_EQ(config.get<bool>("has_projection"), true);
}

TEST_F(ResidualBlockTest, Clone) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale");

  ResidualBlock<float> original(std::move(main_path), {}, "relu", "test_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "residual_block");
}

TEST_F(ResidualBlockTest, ComputeOutputShape) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale", 3, 3);

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "test_shape");

  std::vector<size_t> input_shape = {1, 3, 32, 32};
  std::vector<size_t> output_shape = residual.compute_output_shape(input_shape);

  // Since main path is just scaling, output shape should match input
  EXPECT_EQ(output_shape, input_shape);
}

// Edge Cases and Numerical Stability

TEST_F(ResidualBlockTest, EdgeCaseZeroGradient) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(2.0f, "scale_2x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "zero_gradient");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  input.fill(1.0f);

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  gradient.fill(0.0f);

  Tensor<float> grad_input(input.shape(), cpu_device_);
  residual.backward(gradient, grad_input);

  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input.data()[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, EdgeCaseLargeValues) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "large_values");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1e6f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: F(x) + x = 1*1e6 + 1e6 = 2e6
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 2e6f, 1e1f);
  }
}

TEST_F(ResidualBlockTest, EdgeCaseNegativeValues) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(-1.0f, "scale_neg");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "negative_values");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = -2.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: F(x) + x = -1*(-2) + (-2) = 2 - 2 = 0
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, NumericalStabilitySmallValues) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "small_values");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1e-6f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: F(x) + x = 1*1e-6 + 1e-6 = 2e-6
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 2e-6f, 1e-12f);
  }
}

TEST_F(ResidualBlockTest, NumericalStabilityBackward) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "backward_stability");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  input.fill(1e-6f);

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  gradient.fill(1e-6f);

  Tensor<float> grad_input(input.shape(), cpu_device_);
  residual.backward(gradient, grad_input);

  // grad_main (1.0 * 1e-6) + grad_shortcut (1e-6) = 2e-6
  const float *grad_input_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 2e-6f, 1e-12f);
  }
}

// Multi-path and Complex Scenarios

TEST_F(ResidualBlockTest, MultiLayerMainPath) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  auto layer1 = create_scaling_layer(0.5f, "scale_1");
  auto layer2 = create_scaling_layer(2.0f, "scale_2");
  main_path.push_back(std::move(layer1[0]));
  main_path.push_back(std::move(layer2[0]));

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "multi_layer");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 2.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: F(x) + x = (2.0 * (0.5 * 2.0)) + 2.0 = (2.0 * 1.0) + 2.0 = 4.0
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 4.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, MultiLayerMainPathBackward) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  auto layer1 = create_scaling_layer(0.5f, "scale_1");
  auto layer2 = create_scaling_layer(2.0f, "scale_2");
  main_path.push_back(std::move(layer1[0]));
  main_path.push_back(std::move(layer2[0]));

  ResidualBlock<float> residual(std::move(main_path), {}, "none", "multi_layer_backward");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  input.fill(1.0f);

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  gradient.fill(1.0f);

  Tensor<float> grad_input(input.shape(), cpu_device_);
  residual.backward(gradient, grad_input);

  // grad_main = 2.0 * 0.5 * 1.0 = 1.0
  // grad_shortcut = 1.0
  // total = 2.0
  const float *grad_input_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 2.0f, 1e-4f);
  }
}

TEST_F(ResidualBlockTest, ReLUNegativeInputSuppressionForward) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(0.0f, "scale_zero");

  ResidualBlock<float> residual(std::move(main_path), {}, "relu", "relu_suppression");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = -1.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  // Expected: relu(F(x) + x) = relu(0 + (-1)) = relu(-1) = 0
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, ReLUNegativeInputSuppressionBackward) {
  std::vector<std::unique_ptr<Layer<float>>> main_path;
  main_path = create_scaling_layer(0.0f, "scale_zero");

  ResidualBlock<float> residual(std::move(main_path), {}, "relu", "relu_suppression_bwd");
  residual.set_device(cpu_device_);
  residual.initialize();

  Tensor<float> input({1, 1, 2, 2}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = -1.0f;
  }

  std::vector<size_t> output_shape = residual.compute_output_shape(input.shape());
  Tensor<float> output(output_shape, cpu_device_);
  residual.forward(input, output);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  gradient.fill(1.0f);

  Tensor<float> grad_input(input.shape(), cpu_device_);
  residual.backward(gradient, grad_input);

  // ReLU blocks gradient when output is negative
  const float *grad_input_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.0f, 1e-5f);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
