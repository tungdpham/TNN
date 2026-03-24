/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Test fixture for ResidualBlock validation tests.
 * These tests verify the mathematical correctness of residual block operations
 * including forward pass (skip connection addition) and backward pass (grad_output distribution).
 */
class ResidualBlockTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    Vec<std::string> device_ids = manager.getAvailableDeviceIDs();

    has_cpu_ = false;

    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.device_type() == DeviceType::CPU) {
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
  void verify_identity_shortcut_forward(const ConstTensor &main_path_output,
                                        const ConstTensor &actual_output,
                                        const std::string &activation_type = "relu",
                                        float tolerance = 1e-5f) {
    EXPECT_EQ(main_path_output->shape(), actual_output->shape());

    const float *main_data = main_path_output->data_as<float>();
    const float *output_data = actual_output->data_as<float>();

    Vec<float> expected_output(actual_output->size());
    for (size_t i = 0; i < actual_output->size(); ++i) {
      // For identity shortcut, F(x) + x
      expected_output[i] = main_data[i];

      // Apply activation
      if (activation_type == "relu") {
        expected_output[i] = std::max(0.0f, expected_output[i]);
      } else if (activation_type == "none" || activation_type == "linear") {
        // No activation
      }
    }

    for (size_t i = 0; i < actual_output->size(); ++i) {
      EXPECT_NEAR(output_data[i], expected_output[i], tolerance)
          << "Mismatch at index " << i << ". Expected: " << expected_output[i]
          << ", Got: " << output_data[i];
    }
  }

  /**
   * Verify backward pass for residual block
   * Gradients are summed from both paths: grad_input = grad_main + grad_shortcut
   */
  void verify_backward_gradient_distribution(const ConstTensor &grad_main,
                                             const ConstTensor &grad_shortcut,
                                             const ConstTensor &actual_grad_input,
                                             float tolerance = 1e-5f) {
    EXPECT_EQ(grad_main->shape(), actual_grad_input->shape());
    EXPECT_EQ(grad_shortcut->shape(), actual_grad_input->shape());

    const float *grad_main_data = grad_main->data_as<float>();
    const float *grad_shortcut_data = grad_shortcut->data_as<float>();
    const float *actual_data = actual_grad_input->data_as<float>();

    for (size_t i = 0; i < actual_grad_input->size(); ++i) {
      float expected = grad_main_data[i] + grad_shortcut_data[i];
      EXPECT_NEAR(actual_data[i], expected, tolerance)
          << "Gradient mismatch at index " << i << ". Expected: " << expected
          << ", Got: " << actual_data[i];
    }
  }

  /**
   * Create a simple linear layer (y = scale * x) for testing
   */
  Vec<std::unique_ptr<Layer>> create_scaling_layer(float scale, const std::string &name = "scale",
                                                   size_t in_channels = 1,
                                                   size_t out_channels = 1) {
    Vec<std::unique_ptr<Layer>> layers;
    // Use a 1x1 conv with specific initialization to act as a simple linear transformation
    auto layer = std::make_unique<LegacyConv2DLayer>(in_channels, out_channels, 1, 1, 1, 1, 0, 0,
                                                     false, name);

    // Set weights to scale value (do this after parameters are initialized)
    auto params = layer->parameters();
    if (!params.empty()) {
      float *weight_data = (params[0])->data_as<float>();
      for (size_t i = 0; i < (params[0])->size(); ++i) {
        weight_data[i] = scale;
      }
    }

    layers.push_back(std::move(layer));
    return layers;
  }

  bool has_cpu_;
};

// Identity Shortcut Tests

TEST_F(ResidualBlockTest, IdentityShortcutForward) {
  // Create simple main path: single layer that multiplies by 2
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(2.0f, "scale_2x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "identity_residual");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1.0f;
  }

  auto output = residual->forward({input})[0];

  EXPECT_EQ(output->shape(), input->shape());

  // Expected: F(x) + x = 2*1 + 1 = 3
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 3.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, IdentityShortcutForwardWithReLU) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(-2.0f, "scale_neg2x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "relu", "identity_relu");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1.0f;
  }

  auto output = residual->forward({input})[0];

  // Expected: relu(F(x) + x) = relu(-2*1 + 1) = relu(-1) = 0
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, IdentityShortcutMultiChannel) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(0.5f, "scale_half", 2, 2);

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "identity_multichannel");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 2, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 8; ++i) {
    input_data[i] = 2.0f;
  }

  auto output = residual->forward({input})[0];

  EXPECT_EQ(output->shape()[0], 1);
  EXPECT_EQ(output->shape()[1], 2);
  EXPECT_EQ(output->shape()[2], 2);
  EXPECT_EQ(output->shape()[3], 2);

  // Expected: F(x) + x where F is conv2d with scale 0.5
  // With 2 input channels and 2 output channels: each output = sum(0.5 * input[i]) + input
  // = (0.5 * 2 + 0.5 * 2) + 2 = 2 + 2 = 4
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 4.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, IdentityShortcutMultiBatch) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "identity_multibatch");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({2, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 8; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  auto output = residual->forward({input})[0];

  EXPECT_EQ(output->shape()[0], 2);
  EXPECT_EQ(output->shape()[1], 1);

  // Expected: F(x) + x = 1*x + x = 2*x
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    float expected = 2.0f * input_data[i];
    EXPECT_NEAR(output_data[i], expected, 1e-5f);
  }
}

// Projection Shortcut Tests

TEST_F(ResidualBlockTest, ProjectionShortcutForward) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(0.5f, "scale_main");

  // Projection shortcut: 1x1 conv with scale 0.25
  Vec<std::unique_ptr<Layer>> shortcut = create_scaling_layer(0.25f, "scale_shortcut");

  auto residual_layer = std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut),
                                                        "none", "projection_residual");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 4.0f;
  }

  auto output = residual->forward({input})[0];

  // Expected: F(x) + shortcut(x) = 0.5*4 + 0.25*4 = 2 + 1 = 3
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 3.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, ProjectionShortcutWithReLU) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(-1.0f, "scale_neg");

  Vec<std::unique_ptr<Layer>> shortcut = create_scaling_layer(0.5f, "scale_short");

  auto residual_layer = std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut),
                                                        "relu", "projection_relu");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 2.0f;
  }

  auto output = residual->forward({input})[0];

  // Expected: relu(F(x) + shortcut(x)) = relu(-1*2 + 0.5*2) = relu(-1) = 0
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

// Backward Pass Tests

TEST_F(ResidualBlockTest, IdentityShortcutBackward) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(2.0f, "scale_2x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "identity_backward");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1.0f;
  }

  auto output = residual->forward({input})[0];

  Tensor grad_output = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *grad_data = grad_output->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    grad_data[i] = 1.0f;
  }

  auto grad_input = residual->backward({grad_output})[0];

  EXPECT_EQ(grad_input->shape(), input->shape());

  // Gradient through linear path + grad_output through shortcut
  // Both contribute equally in identity shortcut: grad = grad_main + grad_shortcut
  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    // grad_main from scaling layer (2.0) * incoming_gradient (1.0) = 2.0
    // grad_shortcut from identity = 1.0
    // Total: 2.0 + 1.0 = 3.0
    EXPECT_NEAR(grad_input_data[i], 3.0f, 1e-4f);
  }
}

TEST_F(ResidualBlockTest, GetConfig) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "relu", "test_residual");
  ResidualBlock *residual = residual_layer.get();

  LayerConfig config = residual->get_config();

  EXPECT_EQ(config.name, "test_residual");
  EXPECT_EQ(config.get<std::string>("activation"), "relu");
  EXPECT_EQ(config.get<bool>("has_projection"), false);
}

TEST_F(ResidualBlockTest, GetConfigWithProjection) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_main");

  Vec<std::unique_ptr<Layer>> shortcut;
  shortcut = create_scaling_layer(0.5f, "scale_short");

  auto residual_layer = std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut),
                                                        "relu", "test_projection");
  ResidualBlock *residual = residual_layer.get();

  LayerConfig config = residual->get_config();

  EXPECT_EQ(config.name, "test_projection");
  EXPECT_EQ(config.get<bool>("has_projection"), true);
}

TEST_F(ResidualBlockTest, ComputeOutputShape) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale", 3, 3);

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "test_shape");
  ResidualBlock *residual = residual_layer.get();

  Vec<size_t> input_shape = {1, 3, 32, 32};
  Vec<size_t> output_shape = residual->output_shapes({input_shape})[0];

  // Since main path is just scaling, output shape should match input
  EXPECT_EQ(output_shape, input_shape);
}

// Edge Cases and Numerical Stability

TEST_F(ResidualBlockTest, EdgeCaseZeroGradient) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(2.0f, "scale_2x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "zero_gradient");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  input->fill(1.0f);

  auto output = residual->forward({input})[0];

  Tensor grad_output = make_tensor<float>({1, 1, 2, 2}, getHost());
  grad_output->fill(0.0f);

  auto grad_input = residual->backward({grad_output})[0];

  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input->data_as<float>()[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, EdgeCaseLargeValues) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "large_values");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1e6f;
  }

  auto output = residual->forward({input})[0];

  // Expected: F(x) + x = 1*1e6 + 1e6 = 2e6
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 2e6f, 1e1f);
  }
}

TEST_F(ResidualBlockTest, EdgeCaseNegativeValues) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(-1.0f, "scale_neg");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "negative_values");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = -2.0f;
  }

  auto output = residual->forward({input})[0];

  // Expected: F(x) + x = -1*(-2) + (-2) = 2 - 2 = 0
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, NumericalStabilitySmallValues) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "small_values");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 1e-6f;
  }

  auto output = residual->forward({input})[0];

  // Expected: F(x) + x = 1*1e-6 + 1e-6 = 2e-6
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 2e-6f, 1e-12f);
  }
}

TEST_F(ResidualBlockTest, NumericalStabilityBackward) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(1.0f, "scale_1x");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "backward_stability");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  input->fill(1e-6f);

  auto output = residual->forward({input})[0];

  Tensor grad_output = make_tensor<float>({1, 1, 2, 2}, getHost());
  grad_output->fill(1e-6f);

  auto grad_input = residual->backward({grad_output})[0];

  // grad_main (1.0 * 1e-6) + grad_shortcut (1e-6) = 2e-6
  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 2e-6f, 1e-12f);
  }
}

// Multi-path and Complex Scenarios

TEST_F(ResidualBlockTest, MultiLayerMainPath) {
  Vec<std::unique_ptr<Layer>> main_path;
  auto layer1 = create_scaling_layer(0.5f, "scale_1");
  auto layer2 = create_scaling_layer(2.0f, "scale_2");
  main_path.push_back(std::move(layer1[0]));
  main_path.push_back(std::move(layer2[0]));

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "multi_layer");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = 2.0f;
  }

  auto output = residual->forward({input})[0];

  // Expected: F(x) + x = (2.0 * (0.5 * 2.0)) + 2.0 = (2.0 * 1.0) + 2.0 = 4.0
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 4.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, MultiLayerMainPathBackward) {
  Vec<std::unique_ptr<Layer>> main_path;
  auto layer1 = create_scaling_layer(0.5f, "scale_1");
  auto layer2 = create_scaling_layer(2.0f, "scale_2");
  main_path.push_back(std::move(layer1[0]));
  main_path.push_back(std::move(layer2[0]));

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "none", "multi_layer_backward");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  input->fill(1.0f);

  auto output = residual->forward({input})[0];

  Tensor grad_output = make_tensor<float>({1, 1, 2, 2}, getHost());
  grad_output->fill(1.0f);

  auto grad_input = residual->backward({grad_output})[0];

  // grad_main = 2.0 * 0.5 * 1.0 = 1.0
  // grad_shortcut = 1.0
  // total = 2.0
  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 2.0f, 1e-4f);
  }
}

TEST_F(ResidualBlockTest, ReLUNegativeInputSuppressionForward) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(0.0f, "scale_zero");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "relu", "relu_suppression");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = -1.0f;
  }

  auto output = residual->forward({input})[0];

  // Expected: relu(F(x) + x) = relu(0 + (-1)) = relu(-1) = 0
  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(ResidualBlockTest, ReLUNegativeInputSuppressionBackward) {
  Vec<std::unique_ptr<Layer>> main_path;
  main_path = create_scaling_layer(0.0f, "scale_zero");

  auto residual_layer = std::make_unique<ResidualBlock>(
      std::move(main_path), Vec<std::unique_ptr<Layer>>{}, "relu", "relu_suppression_bwd");
  ResidualBlock *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(residual_layer));
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({1, 1, 2, 2}, getHost());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 4; ++i) {
    input_data[i] = -1.0f;
  }

  auto output = residual->forward({input})[0];

  Tensor grad_output = make_tensor<float>({1, 1, 2, 2}, getHost());
  grad_output->fill(1.0f);

  auto grad_input = residual->backward({grad_output})[0];

  // ReLU blocks grad_output when output is negative
  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.0f, 1e-5f);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
