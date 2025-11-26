/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

/**
 * Test fixture for DenseLayer validation tests.
 * These tests verify the mathematical correctness of fully connected layer operations
 * including forward and backward passes.
 */
class DenseLayerTest : public ::testing::Test {
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

  // Verify forward pass output shape
  void verify_output_shape(const Tensor<float> &input, const Tensor<float> &output,
                           size_t output_features) {
    size_t batch_size = input.batch_size();

    EXPECT_EQ(output.batch_size(), batch_size);
    EXPECT_EQ(output.channels(), output_features);
    EXPECT_EQ(output.height(), 1);
    EXPECT_EQ(output.width(), 1);
  }

  // Verify forward pass numerical correctness
  void verify_forward_result(const Tensor<float> &input, const Tensor<float> &output,
                             const Tensor<float> &weights, const Tensor<float> *bias,
                             float tolerance = 1e-4f) {
    const float *input_data = input.data();
    const float *output_data = output.data();
    const float *weight_data = weights.data();
    const float *bias_data = bias ? bias->data() : nullptr;

    size_t batch_size = input.batch_size();
    size_t input_features = input.channels();
    size_t output_features = output.channels();

    // For each batch and output feature, compute expected value via matrix multiplication
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t out_f = 0; out_f < output_features; ++out_f) {
        float expected = bias_data ? bias_data[out_f] : 0.0f;

        // Compute dot product: weights[out_f, :] · input[n, :]
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

  // Verify backward pass gradient shape
  void verify_gradient_shape(const Tensor<float> &gradient, const Tensor<float> &grad_input,
                             const Tensor<float> &original_input) {
    EXPECT_EQ(grad_input.shape(), original_input.shape());
  }

  // Verify backward pass numerical correctness for input gradients
  void verify_backward_result(const Tensor<float> &grad_output, const Tensor<float> &grad_input,
                              const Tensor<float> &weights, float tolerance = 1e-4f) {
    const float *grad_output_data = grad_output.data();
    const float *grad_input_data = grad_input.data();
    const float *weight_data = weights.data();

    size_t batch_size = grad_input.batch_size();
    size_t input_features = grad_input.channels();
    size_t output_features = grad_output.channels();

    // For each batch and input feature, compute expected gradient
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t in_f = 0; in_f < input_features; ++in_f) {
        float expected_grad = 0.0f;

        // Compute: weights[:, in_f]^T · grad_output[n, :]
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

// Forward Pass Tests

TEST_F(DenseLayerTest, BasicForwardPass) {
  DenseLayer<float> layer(10, 5, true, "test_dense");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 5);
  EXPECT_EQ(output.batch_size(), 2);
  EXPECT_EQ(output.channels(), 5);

  // Verify numerical correctness
  auto params = layer.parameters();
  verify_forward_result(input, output, *params[0], params.size() > 1 ? params[1] : nullptr);
}

TEST_F(DenseLayerTest, ForwardPassSingleBatch) {
  DenseLayer<float> layer(20, 10, true, "test_dense_single");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 20, 1, 1}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 10);
  EXPECT_EQ(output.batch_size(), 1);
  EXPECT_EQ(output.channels(), 10);
}

TEST_F(DenseLayerTest, ForwardPassMultiBatch) {
  DenseLayer<float> layer(15, 8, false, "test_dense_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({4, 15, 1, 1}, cpu_device_);
  input.fill(0.5f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 8);
  EXPECT_EQ(output.batch_size(), 4);
  EXPECT_EQ(output.channels(), 8);
}

TEST_F(DenseLayerTest, ForwardPassLargeLayer) {
  DenseLayer<float> layer(128, 64, true, "test_dense_large");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 128, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 64);
  EXPECT_EQ(output.channels(), 64);
}

TEST_F(DenseLayerTest, ForwardPassWithBias) {
  DenseLayer<float> layer(10, 5, true, "test_dense_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 5);
  EXPECT_EQ(output.channels(), 5);
}

TEST_F(DenseLayerTest, ForwardPassWithoutBias) {
  DenseLayer<float> layer(10, 5, false, "test_dense_no_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 5);
  EXPECT_EQ(output.channels(), 5);
}

TEST_F(DenseLayerTest, ForwardPassVariableInput) {
  DenseLayer<float> layer(6, 3, true, "test_dense_variable");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 6, 1, 1}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 5);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 3);
}

// Backward Pass Tests

TEST_F(DenseLayerTest, BasicBackwardPass) {
  DenseLayer<float> layer(10, 5, true, "test_dense_backward");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.channels(), 10);

  // Verify numerical correctness
  auto params = layer.parameters();
  verify_backward_result(gradient, grad_input, *params[0]);
}

TEST_F(DenseLayerTest, BackwardPassSingleBatch) {
  DenseLayer<float> layer(20, 10, true, "test_dense_backward_single");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 20, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.batch_size(), 1);
}

TEST_F(DenseLayerTest, BackwardPassMultiBatch) {
  DenseLayer<float> layer(15, 8, false, "test_dense_backward_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({4, 15, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.batch_size(), 4);
}

TEST_F(DenseLayerTest, BackwardPassVariableGradient) {
  DenseLayer<float> layer(8, 4, true, "test_dense_backward_var");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 8, 1, 1}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  float *grad_data = gradient.data();
  for (size_t i = 0; i < gradient.size(); ++i) {
    grad_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(DenseLayerTest, BackwardPassWithBias) {
  DenseLayer<float> layer(10, 5, true, "test_dense_backward_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(DenseLayerTest, BackwardPassWithoutBias) {
  DenseLayer<float> layer(10, 5, false, "test_dense_backward_no_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
}

// Configuration Tests

TEST_F(DenseLayerTest, ComputeOutputShape) {
  DenseLayer<float> layer(128, 64, true, "test_dense_shape");

  std::vector<size_t> input_shape = {2, 128, 1, 1};
  std::vector<size_t> expected_shape = {2, 64, 1, 1};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(DenseLayerTest, GetConfig) {
  DenseLayer<float> layer(100, 50, true, "test_dense_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_dense_config");
  EXPECT_EQ(config.get<size_t>("input_features"), 100);
  EXPECT_EQ(config.get<size_t>("output_features"), 50);
  EXPECT_EQ(config.get<bool>("use_bias"), true);
}

TEST_F(DenseLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_dense_recreate";
  config.parameters["input_features"] = size_t(64);
  config.parameters["output_features"] = size_t(32);
  config.parameters["use_bias"] = true;

  auto layer = DenseLayer<float>::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  EXPECT_EQ(layer->type(), "dense");
}

TEST_F(DenseLayerTest, Clone) {
  DenseLayer<float> original(100, 50, true, "test_dense_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "dense");
  EXPECT_EQ(cloned->type(), original.type());
}

// Edge Cases

TEST_F(DenseLayerTest, EdgeCaseSmallLayer) {
  DenseLayer<float> layer(2, 1, true, "test_small_layer");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 2, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  EXPECT_EQ(output.channels(), 1);
  EXPECT_EQ(output.batch_size(), 1);
}

TEST_F(DenseLayerTest, EdgeCaseZeroGradient) {
  DenseLayer<float> layer(10, 5, true, "test_zero_gradient");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(0.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(DenseLayerTest, EdgeCaseLargeValues) {
  DenseLayer<float> layer(10, 5, false, "test_large_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1e6f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 5);
}

TEST_F(DenseLayerTest, EdgeCaseNegativeValues) {
  DenseLayer<float> layer(8, 4, true, "test_negative_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 8, 1, 1}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 4);
}

TEST_F(DenseLayerTest, EdgeCaseLargeBatch) {
  DenseLayer<float> layer(20, 10, true, "test_large_batch");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({32, 20, 1, 1}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 10);
  EXPECT_EQ(output.batch_size(), 32);
}

// Numerical Stability Tests

TEST_F(DenseLayerTest, NumericalStabilitySmallValues) {
  DenseLayer<float> layer(10, 5, true, "test_small_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1e-6f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 5);
}

TEST_F(DenseLayerTest, BackwardNumericalStability) {
  DenseLayer<float> layer(10, 5, false, "test_backward_stability");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  input.fill(1e-6f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1e-6f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(DenseLayerTest, NumericalStabilityMixedValues) {
  DenseLayer<float> layer(10, 5, true, "test_mixed_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 10, 1, 1}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 5);
}

// Multiple Pass Tests

TEST_F(DenseLayerTest, MultipleForwardBackwardPasses) {
  DenseLayer<float> layer(10, 5, true, "test_multiple_passes");
  layer.set_device(cpu_device_);
  layer.initialize();

  // First pass
  Tensor<float> input1({2, 10, 1, 1}, cpu_device_);
  input1.fill(1.0f);
  const Tensor<float> &output1 = layer.forward(input1);
  Tensor<float> gradient1(output1.shape(), cpu_device_);
  gradient1.fill(1.0f);
  layer.backward(gradient1);

  // Second pass
  Tensor<float> input2({2, 10, 1, 1}, cpu_device_);
  input2.fill(2.0f);
  const Tensor<float> &output2 = layer.forward(input2);
  Tensor<float> gradient2(output2.shape(), cpu_device_);
  gradient2.fill(1.0f);
  const Tensor<float> &grad_input2 = layer.backward(gradient2);

  verify_gradient_shape(gradient2, grad_input2, input2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
