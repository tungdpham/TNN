/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace tnn;

/**
 * Test fixture for LayerBuilder residual block tests.
 * These tests verify that the LayerBuilder correctly constructs
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
  void verify_output_values(const Tensor &output, float expected_min, float expected_max,
                            const std::string &test_name = "") {
    const float *output_data = output->data_as<float>();
    for (size_t i = 0; i < output->size(); ++i) {
      EXPECT_GE(output_data[i], expected_min)
          << test_name << ": Value at index " << i << " below minimum";
      EXPECT_LE(output_data[i], expected_max)
          << test_name << ": Value at index " << i << " above maximum";
    }
  }

  bool has_cpu_;
  const Device *cpu_device_;
};

TEST_F(SequentialResidualBlockTest, BasicResidualBlockIdentityShortcut) {

  auto layers =
      LayerBuilder().input({64, 32, 32}).basic_residual_block(64, 64, 1, "basic_64_64").build();
  Sequential model("test_basic_identity", std::move(layers));

  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 64, 32, 32}, "BasicResidualBlockIdentityShortcut");

  const float *output_data = output->data_as<float>();
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
      LayerBuilder().input({64, 32, 32}).basic_residual_block(64, 128, 1, "basic_64_128").build();

  auto model = Sequential("test_basic_projection", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 128, 32, 32}, "BasicResidualBlockProjectionShortcut");

  const float *output_data = output->data_as<float>();
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
                    .input({64, 32, 32})
                    .basic_residual_block(64, 64, 2, "basic_64_64_stride2")
                    .build();

  auto model = Sequential("test_basic_strided", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 64, 16, 16}, "BasicResidualBlockStridedShortcut");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockStridedAndProjection) {

  auto layers = LayerBuilder()
                    .input({64, 32, 32})
                    .basic_residual_block(64, 128, 2, "basic_64_128_stride2")
                    .build();

  auto model = Sequential("test_basic_strided_projection", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 128, 16, 16}, "BasicResidualBlockStridedAndProjection");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockBackward) {

  auto layers =
      LayerBuilder().input({32, 16, 16}).basic_residual_block(32, 32, 1, "basic_32_32").build();

  auto model = Sequential("test_basic_backward", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 32, 16, 16}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);
  EXPECT_EQ(output->shape(), input->shape());

  Tensor grad_output = Tensor::create<float>({1, 32, 16, 16}, cpu_device_);
  grad_output->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  model.backward(grad_output, grad_input);

  verify_output_shape(grad_input->shape(), input->shape(), "BasicResidualBlockBackward");

  const float *grad_data = grad_input->data_as<float>();
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
                    .input({64, 32, 32})
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 128, 2)
                    .basic_residual_block(128, 128, 1)
                    .build();

  auto model = Sequential("test_basic_multiple", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 128, 16, 16}, "BasicResidualBlockMultipleBlocks");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockIdentityShortcut) {

  auto layers =
      LayerBuilder().input({256, 32, 32}).bottleneck_residual_block(256, 64, 256, 1).build();

  auto model = Sequential("test_bottleneck_identity", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 256, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 256, 32, 32}, "BottleneckResidualBlockIdentityShortcut");

  const float *output_data = output->data_as<float>();
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
      LayerBuilder().input({64, 32, 32}).bottleneck_residual_block(64, 64, 256, 1).build();

  auto model = Sequential("test_bottleneck_projection", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 256, 32, 32},
                      "BottleneckResidualBlockProjectionShortcut");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockStridedShortcut) {

  auto layers =
      LayerBuilder().input({256, 32, 32}).bottleneck_residual_block(256, 64, 256, 2).build();

  auto model = Sequential("test_bottleneck_strided", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 256, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 256, 16, 16}, "BottleneckResidualBlockStridedShortcut");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockStridedAndProjection) {

  auto layers =
      LayerBuilder().input({64, 32, 32}).bottleneck_residual_block(64, 64, 256, 2).build();

  auto model = Sequential("test_bottleneck_strided_projection", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 256, 16, 16},
                      "BottleneckResidualBlockStridedAndProjection");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockBackward) {

  auto layers = LayerBuilder().input({64, 16, 16}).bottleneck_residual_block(64, 32, 64, 1).build();

  auto model = Sequential("test_bottleneck_backward", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 16, 16}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);
  EXPECT_EQ(output->shape(), input->shape());

  Tensor grad_output = Tensor::create<float>({1, 64, 16, 16}, cpu_device_);
  grad_output->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  model.backward(grad_output, grad_input);

  verify_output_shape(grad_input->shape(), input->shape(), "BottleneckResidualBlockBackward");

  const float *grad_data = grad_input->data_as<float>();
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
                    .input({64, 32, 32})
                    .bottleneck_residual_block(64, 64, 256, 1)
                    .bottleneck_residual_block(256, 64, 256, 1)
                    .bottleneck_residual_block(256, 128, 512, 2)
                    .bottleneck_residual_block(512, 128, 512, 1)
                    .build();

  auto model = Sequential("test_bottleneck_multiple", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 512, 16, 16}, "BottleneckResidualBlockMultipleBlocks");
}

TEST_F(SequentialResidualBlockTest, MixedBasicAndBottleneckBlocks) {

  auto layers = LayerBuilder()
                    .input({64, 32, 32})
                    .basic_residual_block(64, 64, 1)
                    .bottleneck_residual_block(64, 64, 256, 1)
                    .basic_residual_block(256, 256, 1)
                    .bottleneck_residual_block(256, 128, 512, 2)
                    .build();

  auto model = Sequential("test_mixed_blocks", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 512, 16, 16}, "MixedBasicAndBottleneckBlocks");
}

TEST_F(SequentialResidualBlockTest, ResNet18LikeArchitecture) {

  auto layers = LayerBuilder()
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

  auto model = Sequential("test_resnet18_like", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 512, 4, 4}, "ResNet18LikeArchitecture");
}

TEST_F(SequentialResidualBlockTest, ResNet50LikeArchitecture) {

  auto layers = LayerBuilder()
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

  auto model = Sequential("test_resnet50_like", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 64, 32, 32}, cpu_device_);
  input->fill(1.0f);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);

  verify_output_shape(output->shape(), {1, 1024, 8, 8}, "ResNet50LikeArchitecture");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockOutputShapeComputation) {

  auto layers = LayerBuilder().input({64, 32, 32}).basic_residual_block(64, 128, 2).build();

  auto model = Sequential("test_basic_output_shape", std::move(layers));
  std::vector<size_t> input_shape = {1, 64, 32, 32};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 128, 16, 16}, "BasicResidualBlockOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockOutputShapeComputation) {

  auto layers =
      LayerBuilder().input({64, 32, 32}).bottleneck_residual_block(64, 64, 256, 2).build();

  auto model = Sequential("test_bottleneck_output_shape", std::move(layers));
  std::vector<size_t> input_shape = {1, 64, 32, 32};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 256, 16, 16},
                      "BottleneckResidualBlockOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, ChainedOutputShapeComputation) {

  auto layers = LayerBuilder()
                    .input({64, 32, 32})
                    .basic_residual_block(64, 64, 1)
                    .basic_residual_block(64, 128, 2)
                    .bottleneck_residual_block(128, 64, 256, 1)
                    .build();

  auto model = Sequential("test_chained_output_shape", std::move(layers));
  std::vector<size_t> input_shape = {1, 64, 32, 32};
  auto output_shape = model.compute_output_shape(input_shape);

  verify_output_shape(output_shape, {1, 256, 16, 16}, "ChainedOutputShapeComputation");
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockGetConfig) {

  auto layers =
      LayerBuilder().input({64, 32, 32}).basic_residual_block(64, 64, 1, "my_basic_block").build();
  Sequential model("test_basic_config", std::move(layers));

  auto config = model.get_config();
  auto json = config.to_json();

  EXPECT_EQ(json["name"], "test_basic_config");
  EXPECT_TRUE(json.contains("layers"));
  EXPECT_GT(json["layers"].size(), 0);
}

TEST_F(SequentialResidualBlockTest, BottleneckResidualBlockGetConfig) {

  auto layers = LayerBuilder()
                    .input({64, 32, 32})
                    .bottleneck_residual_block(64, 64, 256, 1, "my_bottleneck_block")
                    .build();

  Sequential model("test_bottleneck_config", std::move(layers));
  auto config = model.get_config().to_json();

  EXPECT_EQ(config["name"], "test_bottleneck_config");
  EXPECT_TRUE(config.contains("layers"));
  EXPECT_GT(config["layers"].size(), 0);
}

TEST_F(SequentialResidualBlockTest, BasicResidualBlockNumericalStability) {

  auto layers = LayerBuilder().input({16, 8, 8}).basic_residual_block(16, 16, 1).build();
  Sequential model("test_basic_numerical", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  std::vector<float> scales = {0.01f, 0.1f, 1.0f, 10.0f};
  for (float scale : scales) {
    Tensor input = Tensor::create<float>({1, 16, 8, 8}, cpu_device_);
    float *input_data = input->data_as<float>();
    srand(42);
    for (size_t i = 0; i < input->size(); ++i) {
      input_data[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }

    Tensor output = Tensor::create<float>({}, cpu_device_);
    model.forward(input, output);

    const float *output_data = output->data_as<float>();
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

  auto layers = LayerBuilder().input({32, 8, 8}).bottleneck_residual_block(32, 16, 32, 1).build();
  Sequential model("test_bottleneck_numerical", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  std::vector<float> scales = {0.01f, 0.1f, 1.0f, 10.0f};
  for (float scale : scales) {
    Tensor input = Tensor::create<float>({1, 32, 8, 8}, cpu_device_);
    float *input_data = input->data_as<float>();
    srand(42);
    for (size_t i = 0; i < input->size(); ++i) {
      input_data[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }

    Tensor output = Tensor::create<float>({}, cpu_device_);
    model.forward(input, output);

    const float *output_data = output->data_as<float>();
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
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 8, 8, 8}, cpu_device_);
  srand(42);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);
  Tensor grad_output = Tensor::create<float>(output->shape(), cpu_device_);
  grad_output->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  model.backward(grad_output, grad_input);

  const float *grad_data = grad_input->data_as<float>();
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

  auto layers = LayerBuilder().input({16, 8, 8}).bottleneck_residual_block(16, 8, 16, 1).build();
  Sequential model("test_bottleneck_gradient_finite", std::move(layers));
  model.set_device(*cpu_device_);
  model.init();

  Tensor input = Tensor::create<float>({1, 16, 8, 8}, cpu_device_);
  srand(42);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);
  Tensor grad_output = Tensor::create<float>(output->shape(), cpu_device_);
  grad_output->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  model.backward(grad_output, grad_input);

  const float *grad_data = grad_input->data_as<float>();
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
  model.set_device(*cpu_device_);
  model.set_seed(123);
  model.init();

  Tensor input = Tensor::create<float>({1, 16, 16, 16}, cpu_device_);
  input->fill_random_uniform(0.0f, 0.1f, 456);

  Tensor output = Tensor::create<float>({}, cpu_device_);
  model.forward(input, output);
  Tensor grad_output = Tensor::create<float>(output->shape(), cpu_device_);
  grad_output->fill(0.001f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  model.backward(grad_output, grad_input);

  const float *grad_data = grad_input->data_as<float>();
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
