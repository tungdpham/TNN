/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "device/device_manager.hpp"
#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Test fixture for LayerBuilder residual block tests.
 * These tests verify that the LayerBuilder correctly constructs
 * basic and bottleneck residual blocks with proper dimensions and forward/backward passes.
 */
class SequentialResidualBlockTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {}

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
  void verify_output_values(const ConstTensor &output, float expected_min, float expected_max,
                            const std::string &test_name = "") {
    Tensor output_cpu = output->to_cpu();
    const float *output_data = output_cpu->data_as<float>();
    for (size_t i = 0; i < output->size(); ++i) {
      EXPECT_GE(output_data[i], expected_min)
          << test_name << ": Value at index " << i << " below minimum";
      EXPECT_LE(output_data[i], expected_max)
          << test_name << ": Value at index " << i << " above maximum";
    }
  }
};

TEST_F(SequentialResidualBlockTest, BasicResidualBlockIdentityShortcut) {
  auto layers =
      LayerBuilder().input({32, 32, 64}).basic_residual_block(64, 64, 1, "basic_64_64").build();
  Sequential model("test_basic_identity", std::move(layers));

  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 32, 32, 64}, "BasicResidualBlockIdentityShortcut");

  Tensor output_cpu = output->to_cpu();
  const float *output_data = output_cpu->data_as<float>();
  bool has_nonzero = false;
  for (size_t i = 0; i < output->size(); ++i) {
    if (output_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockProjectionShortcut) {
  auto layers =
      LayerBuilder().input({32, 32, 64}).basic_residual_block(64, 128, 1, "basic_64_128").build();

  auto model = Sequential("test_basic_projection", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 32, 32, 128}, "BasicResidualBlockProjectionShortcut");

  Tensor output_cpu = output->to_cpu();
  const float *output_data = output_cpu->data_as<float>();
  bool has_nonzero = false;
  for (size_t i = 0; i < output->size(); ++i) {
    if (output_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockStridedShortcut) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .basic_residual_block(64, 64, 2, "basic_64_64_stride2")
                    .build();

  auto model = Sequential("test_basic_strided", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 64}, "BasicResidualBlockStridedShortcut");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockStridedAndProjection) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .basic_residual_block(64, 128, 2, "basic_64_128_stride2")
                    .build();

  auto model = Sequential("test_basic_strided_projection", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 128}, "BasicResidualBlockStridedAndProjection");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockBackward) {
  auto layers =
      LayerBuilder().input({16, 16, 32}).basic_residual_block(32, 32, 1, "basic_32_32").build();

  auto model = Sequential("test_basic_backward", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 16, 16, 32}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);
  EXPECT_EQ(output->shape(), input->shape());

  Tensor grad_output = make_tensor<float>({1, 16, 16, 32}, getGPU());
  grad_output->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getGPU());
  model.backward(grad_output, grad_input);

  verify_output_shape(grad_input->shape(), input->shape(), "BasicResidualBlockBackward");

  Tensor grad_input_cpu = grad_input->to_cpu();
  const float *grad_data = grad_input_cpu->data_as<float>();
  bool has_nonzero = false;
  for (size_t i = 0; i < grad_input->size(); ++i) {
    if (grad_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockMultipleBlocks) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 128, 2)
                    .basic_residual_block(128, 128, 1)
                    .build();

  auto model = Sequential("test_basic_multiple", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 128}, "BasicResidualBlockMultipleBlocks");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockIdentityShortcut) {
  auto layers =
      LayerBuilder().input({32, 32, 256}).bottleneck_residual_block(256, 64, 256, 1).build();

  auto model = Sequential("test_bottleneck_identity", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 256}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 32, 32, 256}, "BottleneckResidualBlockIdentityShortcut");

  Tensor output_cpu = output->to_cpu();
  const float *output_data = output_cpu->data_as<float>();
  bool has_nonzero = false;
  for (size_t i = 0; i < output->size(); ++i) {
    if (output_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockProjectionShortcut) {
  auto layers =
      LayerBuilder().input({32, 32, 64}).bottleneck_residual_block(64, 64, 256, 1).build();

  auto model = Sequential("test_bottleneck_projection", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 32, 32, 256},
                      "BottleneckResidualBlockProjectionShortcut");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockStridedShortcut) {
  auto layers =
      LayerBuilder().input({32, 32, 256}).bottleneck_residual_block(256, 64, 256, 2).build();

  auto model = Sequential("test_bottleneck_strided", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 256}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 256}, "BottleneckResidualBlockStridedShortcut");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockStridedAndProjection) {
  auto layers =
      LayerBuilder().input({32, 32, 64}).bottleneck_residual_block(64, 64, 256, 2).build();

  auto model = Sequential("test_bottleneck_strided_projection", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 256},
                      "BottleneckResidualBlockStridedAndProjection");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockBackward) {
  auto layers = LayerBuilder().input({16, 16, 64}).bottleneck_residual_block(64, 32, 64, 1).build();

  auto model = Sequential("test_bottleneck_backward", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 16, 16, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);
  EXPECT_EQ(output->shape(), input->shape());

  Tensor grad_output = make_tensor<float>({1, 16, 16, 64}, getGPU());
  grad_output->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getGPU());
  model.backward(grad_output, grad_input);

  verify_output_shape(grad_input->shape(), input->shape(), "BottleneckResidualBlockBackward");

  Tensor grad_input_cpu = grad_input->to_cpu();
  const float *grad_data = grad_input_cpu->data_as<float>();
  bool has_nonzero = false;
  for (size_t i = 0; i < grad_input->size(); ++i) {
    if (grad_data[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockMultipleBlocks) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .bottleneck_residual_block(64, 64, 256, 1)
                    .bottleneck_residual_block(256, 64, 256, 1)
                    .bottleneck_residual_block(256, 128, 512, 2)
                    .bottleneck_residual_block(512, 128, 512, 1)
                    .build();

  auto model = Sequential("test_bottleneck_multiple", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 512}, "BottleneckResidualBlockMultipleBlocks");
}

TEST_F(SequentialResidualBlockTest, MixedBasicAndBottleneckBlocks) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .basic_residual_block(64, 64, 1)
                    .bottleneck_residual_block(64, 64, 256, 1)
                    .basic_residual_block(256, 256, 1)
                    .bottleneck_residual_block(256, 128, 512, 2)
                    .build();

  auto model = Sequential("test_mixed_blocks", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 16, 16, 512}, "MixedBasicAndBottleneckBlocks");
}

TEST_F(SequentialResidualBlockTest, ResNet18LikeArchitecture) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 128, 2)
                    .basic_residual_block(128, 128, 1)
                    .basic_residual_block(128, 256, 2)
                    .basic_residual_block(256, 256, 1)
                    .basic_residual_block(256, 512, 2)
                    .basic_residual_block(512, 512, 1)
                    .build();

  auto model = Sequential("test_resnet18_like", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 4, 4, 512}, "ResNet18LikeArchitecture");
}

TEST_F(SequentialResidualBlockTest, ResNet50LikeArchitecture) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
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

  auto model = Sequential("test_resnet50_like", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 32, 32, 64}, getGPU());
  input->fill(1.0f);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 8, 8, 1024}, "ResNet50LikeArchitecture");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockOutputShapeComputation) {
  auto layers = LayerBuilder().input({32, 32, 64}).basic_residual_block(64, 128, 2).build();

  auto model = Sequential("test_basic_output_shape", std::move(layers));
  std::vector<size_t> input_shape = {1, 32, 32, 64};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 16, 16, 128}, "BasicResidualBlockOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockOutputShapeComputation) {
  auto layers =
      LayerBuilder().input({32, 32, 64}).bottleneck_residual_block(64, 64, 256, 2).build();

  auto model = Sequential("test_bottleneck_output_shape", std::move(layers));
  std::vector<size_t> input_shape = {1, 32, 32, 64};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 16, 16, 256},
                      "BottleneckResidualBlockOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, ChainedOutputShapeComputation) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 128, 2)
                    .bottleneck_residual_block(128, 64, 256, 1)
                    .build();

  auto model = Sequential("test_chained_output_shape", std::move(layers));
  std::vector<size_t> input_shape = {1, 32, 32, 64};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 16, 16, 256}, "ChainedOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockGetConfig) {
  auto layers =
      LayerBuilder().input({32, 32, 64}).basic_residual_block(64, 64, 1, "my_basic_block").build();
  Sequential model("test_basic_config", std::move(layers));

  auto config = model.get_config();
  auto json = config.to_json();

  EXPECT_EQ(json["name"], "test_basic_config");
  EXPECT_TRUE(json["parameters"].contains("layers"));
  EXPECT_GT(json["parameters"]["layers"].size(), 0);
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockGetConfig) {
  auto layers = LayerBuilder()
                    .input({32, 32, 64})
                    .bottleneck_residual_block(64, 64, 256, 1, "my_bottleneck_block")
                    .build();

  Sequential model("test_bottleneck_config", std::move(layers));
  auto config = model.get_config().to_json();

  EXPECT_EQ(config["name"], "test_bottleneck_config");
  EXPECT_TRUE(config["parameters"].contains("layers"));
  EXPECT_GT(config["parameters"]["layers"].size(), 0);
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockNumericalStability) {
  auto layers = LayerBuilder().input({8, 8, 16}).basic_residual_block(16, 16, 1).build();
  Sequential model("test_basic_numerical", std::move(layers));
  model.set_device(getGPU());
  model.init();

  std::vector<float> scales = {0.01f, 0.1f, 1.0f, 10.0f};
  for (float scale : scales) {
    Tensor input = make_tensor<float>({1, 8, 8, 16}, getCPU());
    float *input_data = input->data_as<float>();
    srand(42);
    for (size_t i = 0; i < input->size(); ++i) {
      input_data[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }
    input = input->to_gpu();

    Tensor output = make_tensor<float>({}, getGPU());
    model.forward(input, output);

    Tensor output_cpu = output->to_cpu();
    const float *output_data = output_cpu->data_as<float>();
    for (size_t i = 0; i < output->size(); ++i) {
      EXPECT_TRUE(std::isfinite(output_data[i]))
          << "Output contains non-finite value at index " << i << " with scale " << scale;
    }

    // float output_sum = 0.0f;
    float output_abs_max = 0.0f;
    for (size_t i = 0; i < output->size(); ++i) {
      // output_sum += output_data[i];
      output_abs_max = std::max(output_abs_max, std::abs(output_data[i]));
    }
    EXPECT_GT(output_abs_max, 0.0f) << "Output should have non-zero values with scale " << scale;
  }
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockNumericalStability) {
  auto layers = LayerBuilder().input({8, 8, 32}).bottleneck_residual_block(32, 16, 32, 1).build();
  Sequential model("test_bottleneck_numerical", std::move(layers));
  model.set_device(getGPU());
  model.init();

  std::vector<float> scales = {0.01f, 0.1f, 1.0f, 10.0f};
  for (float scale : scales) {
    Tensor input = make_tensor<float>({1, 8, 8, 32}, getCPU());
    float *input_data = input->data_as<float>();
    srand(42);
    for (size_t i = 0; i < input->size(); ++i) {
      input_data[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }
    input = input->to_gpu();

    Tensor output = make_tensor<float>({}, getGPU());
    model.forward(input, output);

    Tensor output_cpu = output->to_cpu();
    const float *output_data = output_cpu->data_as<float>();
    for (size_t i = 0; i < output->size(); ++i) {
      EXPECT_TRUE(std::isfinite(output_data[i]))
          << "Output contains non-finite value at index " << i << " with scale " << scale;
    }

    float output_abs_max = 0.0f;
    for (size_t i = 0; i < output->size(); ++i) {
      output_abs_max = std::max(output_abs_max, std::abs(output_data[i]));
    }
    EXPECT_GT(output_abs_max, 0.0f) << "Output should have non-zero values with scale " << scale;
  }
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockGradientFiniteness) {
  auto layers = LayerBuilder().input({8, 8, 8}).basic_residual_block(8, 8, 1).build();
  Sequential model("test_basic_gradient_finite", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 8, 8, 8}, getCPU());
  srand(42);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  input = input->to_gpu();

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);
  Tensor grad_output = make_tensor<float>(output->shape(), getGPU());
  grad_output->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getGPU());
  model.backward(grad_output, grad_input);

  Tensor grad_input_cpu = grad_input->to_cpu();
  const float *grad_data = grad_input_cpu->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_TRUE(std::isfinite(grad_data[i])) << "Gradient contains non-finite value at index " << i;
  }

  bool has_nonzero_grad = false;
  for (size_t i = 0; i < grad_input->size(); ++i) {
    if (std::abs(grad_data[i]) > 1e-6f) {
      has_nonzero_grad = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero_grad) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockGradientFiniteness) {
  auto layers = LayerBuilder().input({8, 8, 16}).bottleneck_residual_block(16, 8, 16, 1).build();
  Sequential model("test_bottleneck_gradient_finite", std::move(layers));
  model.set_device(getGPU());
  model.init();

  Tensor input = make_tensor<float>({1, 8, 8, 16}, getCPU());
  srand(42);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  input = input->to_gpu();

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);
  Tensor grad_output = make_tensor<float>(output->shape(), getGPU());
  grad_output->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getGPU());
  model.backward(grad_output, grad_input);

  Tensor grad_input_cpu = grad_input->to_cpu();
  const float *grad_data = grad_input_cpu->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_TRUE(std::isfinite(grad_data[i])) << "Gradient contains non-finite value at index " << i;
  }

  bool has_nonzero_grad = false;
  for (size_t i = 0; i < grad_input->size(); ++i) {
    if (std::abs(grad_data[i]) > 1e-6f) {
      has_nonzero_grad = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero_grad) << "Gradient should have non-zero values";
}

TEST_F(SequentialResidualBlockTest, ResidualBlockGradientMagnitudes) {
  auto layers = LayerBuilder()
                    .input({16, 16, 16})
                    .basic_residual_block(16, 16, 1)
                    .basic_residual_block(16, 32, 2)
                    .bottleneck_residual_block(32, 16, 32, 1)
                    .build();

  Sequential model("test_gradient_magnitudes", std::move(layers));
  model.set_device(getGPU());
  model.set_seed(123);
  model.init();

  Tensor input = make_tensor<float>({1, 16, 16, 16}, getGPU());
  input->fill_random_uniform(0.0f, 0.1f, 456);

  Tensor output = make_tensor<float>({}, getGPU());
  model.forward(input, output);
  Tensor grad_output = make_tensor<float>(output->shape(), getGPU());
  grad_output->fill(0.001f);

  Tensor grad_input = make_tensor<float>(input->shape(), getGPU());
  model.backward(grad_output, grad_input);

  Tensor grad_input_cpu = grad_input->to_cpu();
  const float *grad_data = grad_input_cpu->data_as<float>();
  float grad_max = -std::numeric_limits<float>::max();
  size_t nonzero_count = 0;

  for (size_t i = 0; i < grad_input->size(); ++i) {
    float abs_grad = std::abs(grad_data[i]);
    if (abs_grad > 1e-10f) {
      grad_max = std::max(grad_max, abs_grad);
      nonzero_count++;
    }
  }

  EXPECT_LT(grad_max, 100.0f) << "Gradient values are too large (exploding gradients)";

  EXPECT_GT(nonzero_count, grad_input->size() * 0.1f)
      << "Too many vanishing gradients: only " << nonzero_count << " non-zero out of "
      << grad_input->size();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
