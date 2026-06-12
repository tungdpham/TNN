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
#include "device/pool_allocator.hpp"
#include "tensor/tensor.hpp"
#include "test_graph_utils.hpp"

using namespace synet;

/**
 * Test fixture for LegacyDenseLayerImpl validation tests.
 * These tests verify the mathematical correctness of fully connected layer operations
 * including forward and backward passes.
 */
class LegacyDenseLayerTest : public ::testing::Test {
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

  void verify_output_shape(const Tensor &input, Tensor &output, size_t output_features) {
    auto input_shape = input.shape();
    auto output_shape = output.shape();
    size_t batch_size = input_shape[0];

    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], output_features);
  }

  void verify_forward_result(const Tensor &input, Tensor &output, const Tensor &weights,
                             const Tensor bias, float tolerance = 1e-4f) {
    const float *input_data = input.data_as<float>();
    const float *output_data = output.data_as<float>();
    const float *weight_data = weights.data_as<float>();
    const float *bias_data = bias ? bias.data_as<float>() : nullptr;

    auto input_shape = input.shape();
    auto output_shape = output.shape();
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

  void verify_gradient_shape(const Tensor &grad_output, Tensor &grad_input,
                             const Tensor &original_input) {
    EXPECT_EQ(grad_input.shape(), original_input.shape());
  }

  void verify_backward_result(const Tensor &grad_output, Tensor &grad_input, const Tensor &weights,
                              float tolerance = 1e-4f) {
    const float *grad_output_data = grad_output.data_as<float>();
    const float *grad_input_data = grad_input.data_as<float>();
    const float *weight_data = weights.data_as<float>();

    auto grad_input_shape = grad_input.shape();
    auto grad_output_shape = grad_output.shape();
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
};

TEST_F(LegacyDenseLayerTest, BasicForwardPass) {
  auto dense_layer = LegacyDenseLayer(10, 5, true, "test_dense");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(dense_layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = dense_layer.forward({input})[0];

  verify_output_shape(input, output, 5);
  auto output_shape = output.shape();
  EXPECT_EQ(output_shape[0], 2);
  EXPECT_EQ(output_shape[1], 5);

  auto params = dense_layer.parameters();
  verify_forward_result(input, output, *params[0], params.size() > 1 ? *params[1] : Tensor());
}

TEST_F(LegacyDenseLayerTest, ForwardPassSingleBatch) {
  auto layer = LegacyDenseLayer(20, 10, true, "test_dense_single");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 20}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 10);
  auto output_shape = output.shape();
  EXPECT_EQ(output_shape[0], 1);
  EXPECT_EQ(output_shape[1], 10);
}

TEST_F(LegacyDenseLayerTest, ForwardPassMultiBatch) {
  auto layer = LegacyDenseLayer(15, 8, false, "test_dense_multibatch");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({4, 15}, DType_t::FP32, getHost());
  input.fill(0.5f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 8);
  auto output_shape = output.shape();
  EXPECT_EQ(output_shape[0], 4);
  EXPECT_EQ(output_shape[1], 8);
}

TEST_F(LegacyDenseLayerTest, ForwardPassLargeLayer) {
  auto layer = LegacyDenseLayer(128, 64, true, "test_dense_large");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 128}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 64);
  auto output_shape = output.shape();
  EXPECT_EQ(output_shape[1], 64);
}

TEST_F(LegacyDenseLayerTest, ForwardPassWithBias) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_dense_bias");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 5);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[1], 5);
}

TEST_F(LegacyDenseLayerTest, ForwardPassWithoutBias) {
  auto layer = LegacyDenseLayer(10, 5, false, "test_dense_no_bias");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 5);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[1], 5);
}

TEST_F(LegacyDenseLayerTest, ForwardPassVariableInput) {
  auto layer = LegacyDenseLayer(6, 3, true, "test_dense_variable");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 6}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 5);
  }

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 3);
}

TEST_F(LegacyDenseLayerTest, BasicBackwardPass) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_dense_backward");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  auto grad_input_shape = grad_input.shape();
  EXPECT_EQ(grad_input_shape[1], 10);

  auto params = layer.parameters();
  verify_backward_result(grad_output, grad_input, *params[0]);
}

TEST_F(LegacyDenseLayerTest, BackwardPassSingleBatch) {
  auto layer = LegacyDenseLayer(20, 10, true, "test_dense_backward_single");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 20}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  auto grad_input_shape = grad_input.shape();
  EXPECT_EQ(grad_input_shape[0], 1);
}

TEST_F(LegacyDenseLayerTest, BackwardPassMultiBatch) {
  auto layer = LegacyDenseLayer(15, 8, false, "test_dense_backward_multibatch");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({4, 15}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  auto grad_input_shape = grad_input.shape();
  EXPECT_EQ(grad_input_shape[0], 4);
}

TEST_F(LegacyDenseLayerTest, BackwardPassVariableGradient) {
  auto layer = LegacyDenseLayer(8, 4, true, "test_dense_backward_var");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  float *grad_data = grad_output.data_as<float>();
  for (size_t i = 0; i < grad_output.size(); ++i) {
    grad_data[i] = static_cast<float>(i + 1);
  }

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(LegacyDenseLayerTest, BackwardPassWithBias) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_dense_backward_bias");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
}

TEST_F(LegacyDenseLayerTest, BackwardPassWithoutBias) {
  auto layer = LegacyDenseLayer(10, 5, false, "test_dense_backward_no_bias");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
}

TEST_F(LegacyDenseLayerTest, ComputeOutputShape) {
  auto layer = LegacyDenseLayer(128, 64, true, "test_dense_shape");

  Vec<size_t> input_shape = {2, 128};
  Vec<size_t> expected_shape = {2, 64};

  Vec<size_t> output_shape = layer.output_shapes({input_shape})[0];

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyDenseLayerTest, GetConfig) {
  auto layer = LegacyDenseLayer(100, 50, true, "test_dense_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_dense_config");
  EXPECT_EQ(config.get<size_t>("input_features"), 100);
  EXPECT_EQ(config.get<size_t>("output_features"), 50);
  EXPECT_EQ(config.get<bool>("use_bias"), true);
}

TEST_F(LegacyDenseLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_dense_recreate";
  config.set("input_features", size_t(64));
  config.set("output_features", size_t(32));
  config.set("use_bias", true);

  auto layer = LegacyDenseLayer::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  EXPECT_EQ(layer.type(), "legacy_dense");
}

TEST_F(LegacyDenseLayerTest, EdgeCaseSmallLayer) {
  auto layer = LegacyDenseLayer(2, 1, true, "test_small_layer");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 2}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[1], 1);
  EXPECT_EQ(out_shape[0], 1);
}

TEST_F(LegacyDenseLayerTest, EdgeCaseZeroGradient) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_zero_gradient");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(0.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
}

TEST_F(LegacyDenseLayerTest, EdgeCaseLargeValues) {
  auto layer = LegacyDenseLayer(10, 5, false, "test_large_values");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1e6f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 5);
}

TEST_F(LegacyDenseLayerTest, EdgeCaseNegativeValues) {
  auto layer = LegacyDenseLayer(8, 4, true, "test_negative_values");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 4);
}

TEST_F(LegacyDenseLayerTest, EdgeCaseLargeBatch) {
  auto layer = LegacyDenseLayer(20, 10, true, "test_large_batch");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({32, 20}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 10);
  auto output_shape = output.shape();
  EXPECT_EQ(output_shape[0], 32);
}

TEST_F(LegacyDenseLayerTest, NumericalStabilitySmallValues) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_small_values");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1e-6f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 5);
}

TEST_F(LegacyDenseLayerTest, BackwardNumericalStability) {
  auto layer = LegacyDenseLayer(10, 5, false, "test_backward_stability");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  input.fill(1e-6f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1e-6f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
}

TEST_F(LegacyDenseLayerTest, NumericalStabilityMixedValues) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_mixed_values");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 10}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
  }

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 5);
}

TEST_F(LegacyDenseLayerTest, MultipleForwardBackwardPasses) {
  auto layer = LegacyDenseLayer(10, 5, true, "test_multiple_passes");

  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input1 = Tensor({2, 10}, DType_t::FP32, getHost());
  input1.fill(1.0f);

  Tensor output1 = layer.forward({input1})[0];

  Tensor gradient1 = Tensor(output1.shape(), DType_t::FP32, getHost());
  gradient1.fill(1.0f);
  Tensor grad_input1 = layer.backward({gradient1})[0];

  Tensor input2 = Tensor({2, 10}, DType_t::FP32, getHost());
  input2.fill(2.0f);
  Tensor output2 = layer.forward({input2})[0];
  Tensor gradient2 = Tensor(output2.shape(), DType_t::FP32, getHost());
  gradient2.fill(1.0f);
  Tensor grad_input2 = layer.backward({gradient2})[0];

  verify_gradient_shape(gradient2, grad_input2, input2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
