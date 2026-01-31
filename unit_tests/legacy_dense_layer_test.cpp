/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/legacy_dense_layer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "device/device_manager.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Test fixture for LegacyDenseLayer validation tests.
 * These tests verify the mathematical correctness of fully connected layer operations
 * including forward and backward passes.
 */
class LegacyLegacyDenseLayerTest : public ::testing::Test {
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

  void verify_output_shape(const Tensor &input, const Tensor &output, size_t output_features) {
    auto input_shape = input->shape();
    auto output_shape = output->shape();
    size_t batch_size = input_shape[0];

    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], output_features);
  }

  void verify_forward_result(const Tensor &input, const Tensor &output, const Tensor &weights,
                             const Tensor bias, float tolerance = 1e-4f) {
    const float *input_data = input->data_as<float>();
    const float *output_data = output->data_as<float>();
    const float *weight_data = weights->data_as<float>();
    const float *bias_data = bias ? bias->data_as<float>() : nullptr;

    auto input_shape = input->shape();
    auto output_shape = output->shape();
    size_t batch_size = input_shape[0];
    size_t input_features = input_shape[1];
    size_t output_features = output_shape[1];

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t out_f = 0; out_f < output_features; ++out_f) {
        float expected = bias_data ? bias_data[out_f] : 0.0f;

        for (size_t in_f = 0; in_f < input_features; ++in_f) {
          size_t input_idx = n * input_features + in_f;
          size_t weight_idx = out_f * input_features + in_f;
          expected += input_data[input_idx] * weight_data[weight_idx];
        }

        size_t output_idx = n * output_features + out_f;
        EXPECT_NEAR(output_data[output_idx], expected, tolerance)
            << "Mismatch at batch=" << n << ", output_feature=" << out_f;
      }
    }
  }

  void verify_gradient_shape(const Tensor &gradient, const Tensor &grad_input,
                             const Tensor &original_input) {
    EXPECT_EQ(grad_input->shape(), original_input->shape());
  }

  void verify_backward_result(const Tensor &grad_output, const Tensor &grad_input,
                              const Tensor &weights, float tolerance = 1e-4f) {
    const float *grad_output_data = grad_output->data_as<float>();
    const float *grad_input_data = grad_input->data_as<float>();
    const float *weight_data = weights->data_as<float>();

    auto grad_input_shape = grad_input->shape();
    auto grad_output_shape = grad_output->shape();
    size_t batch_size = grad_input_shape[0];
    size_t input_features = grad_input_shape[1];
    size_t output_features = grad_output_shape[1];

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t in_f = 0; in_f < input_features; ++in_f) {
        float expected_grad = 0.0f;

        for (size_t out_f = 0; out_f < output_features; ++out_f) {
          size_t grad_output_idx = n * output_features + out_f;
          size_t weight_idx = out_f * input_features + in_f;
          expected_grad += grad_output_data[grad_output_idx] * weight_data[weight_idx];
        }

        size_t grad_input_idx = n * input_features + in_f;
        EXPECT_NEAR(grad_input_data[grad_input_idx], expected_grad, tolerance)
            << "Gradient mismatch at batch=" << n << ", input_feature=" << in_f;
      }
    }
  }

  bool has_cpu_;
  const Device *cpu_device_;
};

TEST_F(LegacyLegacyDenseLayerTest, BasicForwardPass) {
  LegacyDenseLayer layer(10, 5, true, "test_dense");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> expected_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(expected_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 5);
  auto output_shape = output->shape();
  EXPECT_EQ(output_shape[0], 2);
  EXPECT_EQ(output_shape[1], 5);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], params.size() > 1 ? params[1] : nullptr);
}

TEST_F(LegacyLegacyDenseLayerTest, ForwardPassSingleBatch) {
  LegacyDenseLayer layer(20, 10, true, "test_dense_single");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({1, 20}, cpu_device_);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> expected_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(expected_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 10);
  auto output_shape = output->shape();
  EXPECT_EQ(output_shape[0], 1);
  EXPECT_EQ(output_shape[1], 10);
}

TEST_F(LegacyLegacyDenseLayerTest, ForwardPassMultiBatch) {
  LegacyDenseLayer layer(15, 8, false, "test_dense_multibatch");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({4, 15}, cpu_device_);
  input->fill(0.5f);

  std::vector<size_t> expected_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(expected_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 8);
  auto output_shape = output->shape();
  EXPECT_EQ(output_shape[0], 4);
  EXPECT_EQ(output_shape[1], 8);
}

TEST_F(LegacyLegacyDenseLayerTest, ForwardPassLargeLayer) {
  LegacyDenseLayer layer(128, 64, true, "test_dense_large");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 128}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> expected_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(expected_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 64);
  auto output_shape = output->shape();
  EXPECT_EQ(output_shape[1], 64);
}

TEST_F(LegacyLegacyDenseLayerTest, ForwardPassWithBias) {
  LegacyDenseLayer layer(10, 5, true, "test_dense_bias");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({1, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 5);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[1], 5);
}

TEST_F(LegacyLegacyDenseLayerTest, ForwardPassWithoutBias) {
  LegacyDenseLayer layer(10, 5, false, "test_dense_no_bias");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({1, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 5);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[1], 5);
}

TEST_F(LegacyLegacyDenseLayerTest, ForwardPassVariableInput) {
  LegacyDenseLayer layer(6, 3, true, "test_dense_variable");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 6}, cpu_device_);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 5);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 3);
}

TEST_F(LegacyLegacyDenseLayerTest, BasicBackwardPass) {
  LegacyDenseLayer layer(10, 5, true, "test_dense_backward");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[1], 10);

  auto params = layer.parameters();
  verify_backward_result(gradient, grad_input, params[0]);
}

TEST_F(LegacyLegacyDenseLayerTest, BackwardPassSingleBatch) {
  LegacyDenseLayer layer(20, 10, true, "test_dense_backward_single");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({1, 20}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[0], 1);
}

TEST_F(LegacyLegacyDenseLayerTest, BackwardPassMultiBatch) {
  LegacyDenseLayer layer(15, 8, false, "test_dense_backward_multibatch");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({4, 15}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[0], 4);
}

TEST_F(LegacyLegacyDenseLayerTest, BackwardPassVariableGradient) {
  LegacyDenseLayer layer(8, 4, true, "test_dense_backward_var");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 8}, cpu_device_);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  float *grad_data = gradient->data_as<float>();
  for (size_t i = 0; i < gradient->size(); ++i) {
    grad_data[i] = static_cast<float>(i + 1);
  }

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyLegacyDenseLayerTest, BackwardPassWithBias) {
  LegacyDenseLayer layer(10, 5, true, "test_dense_backward_bias");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(LegacyLegacyDenseLayerTest, BackwardPassWithoutBias) {
  LegacyDenseLayer layer(10, 5, false, "test_dense_backward_no_bias");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(LegacyLegacyDenseLayerTest, ComputeOutputShape) {
  LegacyDenseLayer layer(128, 64, true, "test_dense_shape");

  std::vector<size_t> input_shape = {2, 128};
  std::vector<size_t> expected_shape = {2, 64};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyLegacyDenseLayerTest, GetConfig) {
  LegacyDenseLayer layer(100, 50, true, "test_dense_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_dense_config");
  EXPECT_EQ(config.get<size_t>("input_features"), 100);
  EXPECT_EQ(config.get<size_t>("output_features"), 50);
  EXPECT_EQ(config.get<bool>("use_bias"), true);
}

TEST_F(LegacyLegacyDenseLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_dense_recreate";
  config.parameters["input_features"] = size_t(64);
  config.parameters["output_features"] = size_t(32);
  config.parameters["use_bias"] = true;

  auto layer = LegacyDenseLayer::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  EXPECT_EQ(layer->type(), "dense");
}

TEST_F(LegacyLegacyDenseLayerTest, Clone) {
  LegacyDenseLayer original(100, 50, true, "test_dense_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "dense");
  EXPECT_EQ(cloned->type(), original.type());
}

TEST_F(LegacyLegacyDenseLayerTest, EdgeCaseSmallLayer) {
  LegacyDenseLayer layer(2, 1, true, "test_small_layer");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({1, 2}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[1], 1);
  EXPECT_EQ(out_shape[0], 1);
}

TEST_F(LegacyLegacyDenseLayerTest, EdgeCaseZeroGradient) {
  LegacyDenseLayer layer(10, 5, true, "test_zero_gradient");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(0.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(LegacyLegacyDenseLayerTest, EdgeCaseLargeValues) {
  LegacyDenseLayer layer(10, 5, false, "test_large_values");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1e6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 5);
}

TEST_F(LegacyLegacyDenseLayerTest, EdgeCaseNegativeValues) {
  LegacyDenseLayer layer(8, 4, true, "test_negative_values");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({1, 8}, cpu_device_);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 4);
}

TEST_F(LegacyLegacyDenseLayerTest, EdgeCaseLargeBatch) {
  LegacyDenseLayer layer(20, 10, true, "test_large_batch");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({32, 20}, cpu_device_);
  input->fill(1.0f);

  std::vector<size_t> expected_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(expected_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 10);
  auto output_shape = output->shape();
  EXPECT_EQ(output_shape[0], 32);
}

TEST_F(LegacyLegacyDenseLayerTest, NumericalStabilitySmallValues) {
  LegacyDenseLayer layer(10, 5, true, "test_small_values");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1e-6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 5);
}

TEST_F(LegacyLegacyDenseLayerTest, BackwardNumericalStability) {
  LegacyDenseLayer layer(10, 5, false, "test_backward_stability");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  input->fill(1e-6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), cpu_device_);
  gradient->fill(1e-6f);

  Tensor grad_input = Tensor::create<float>(input->shape(), cpu_device_);
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(LegacyLegacyDenseLayerTest, NumericalStabilityMixedValues) {
  LegacyDenseLayer layer(10, 5, true, "test_mixed_values");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input = Tensor::create<float>({2, 10}, cpu_device_);
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, cpu_device_);
  layer.forward(input, output);

  verify_output_shape(input, output, 5);
}

TEST_F(LegacyLegacyDenseLayerTest, MultipleForwardBackwardPasses) {
  LegacyDenseLayer layer(10, 5, true, "test_multiple_passes");
  layer.set_device(*cpu_device_);
  layer.init();

  Tensor input1 = Tensor::create<float>({2, 10}, cpu_device_);
  input1->fill(1.0f);
  std::vector<size_t> output_shape1 = layer.compute_output_shape(input1->shape());
  Tensor output1 = Tensor::create<float>(output_shape1, cpu_device_);
  layer.forward(input1, output1);
  Tensor gradient1 = Tensor::create<float>(output1->shape(), cpu_device_);
  gradient1->fill(1.0f);
  Tensor grad_input1 = Tensor::create<float>(input1->shape(), cpu_device_);
  layer.backward(gradient1, grad_input1);

  Tensor input2 = Tensor::create<float>({2, 10}, cpu_device_);
  input2->fill(2.0f);
  std::vector<size_t> output_shape2 = layer.compute_output_shape(input2->shape());
  Tensor output2 = Tensor::create<float>(output_shape2, cpu_device_);
  layer.forward(input2, output2);
  Tensor gradient2 = Tensor::create<float>(output2->shape(), cpu_device_);
  gradient2->fill(1.0f);
  Tensor grad_input2 = Tensor::create<float>(input2->shape(), cpu_device_);
  layer.backward(gradient2, grad_input2);

  verify_gradient_shape(gradient2, grad_input2, input2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
