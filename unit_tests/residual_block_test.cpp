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
#include "nn/layers.hpp"
#include "test_graph_utils.hpp"
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

  bool has_cpu_;
};

// Identity Shortcut Tests

TEST_F(ResidualBlockTest, IdentityShortcutForward) {
  // Create simple main path: single layer that multiplies by 2
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_2x"));
  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "identity_residual");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(2.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_neg2x"));

  auto residual_layer = ResidualBlock(std::move(main_path), Vec<Layer>{}, "relu", "identity_relu");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(-2.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(2, 2, 1, 1, 1, 1, 0, 0, false, "scale_half"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "identity_multichannel");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(0.5f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_1x"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "identity_multibatch");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(1.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_half"));

  // Projection shortcut: 1x1 conv with scale 0.25
  Vec<Layer> shortcut;
  shortcut.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_quarter"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), std::move(shortcut), "none", "projection_residual");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(0.5f);
  residual->parameters()[1]->fill(0.25f);
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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_neg"));

  Vec<Layer> shortcut;
  shortcut.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_short"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), std::move(shortcut), "relu", "projection_relu");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(-1.0f);
  residual->parameters()[1]->fill(0.5f);
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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_2x"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "identity_backward");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(2.0f);

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

TEST_F(ResidualBlockTest, ComputeOutputShape) {
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(3, 3, 1, 1, 1, 1, 0, 0, false, "scale"));

  auto residual_layer = ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "test_shape");
  ResidualBlockImpl *residual = residual_layer.get();

  Vec<size_t> input_shape = {1, 3, 32, 32};
  Vec<size_t> output_shape = residual->output_shapes({input_shape})[0];

  // Since main path is just scaling, output shape should match input
  EXPECT_EQ(output_shape, input_shape);
}

// Edge Cases and Numerical Stability

TEST_F(ResidualBlockTest, EdgeCaseZeroGradient) {
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_2x"));

  auto residual_layer = ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "zero_gradient");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(2.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_1x"));

  auto residual_layer = ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "large_values");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(1.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_neg"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "negative_values");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(-1.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_1x"));

  auto residual_layer = ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "small_values");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(1.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_1x"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "backward_stability");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(1.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_1"));
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_2"));

  auto residual_layer = ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "multi_layer");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(0.5f);
  residual->parameters()[1]->fill(2.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_1"));
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_2"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "none", "multi_layer_backward");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(0.5f);
  residual->parameters()[1]->fill(2.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_zero"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "relu", "relu_suppression");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(0.0f);

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
  Vec<Layer> main_path;
  main_path.push_back(LegacyConv2DLayer(1, 1, 1, 1, 1, 1, 0, 0, false, "scale_zero"));

  auto residual_layer =
      ResidualBlock(std::move(main_path), Vec<Layer>{}, "relu", "relu_suppression_bwd");
  ResidualBlockImpl *residual = residual_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(residual_layer, allocator);
  residual->parameters()[0]->fill(0.0f);

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
