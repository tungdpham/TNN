/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/legacy_batchnorm_layer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph_builder.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Test fixture for LegacyBatchNormLayer validation tests.
 * These tests verify the mathematical correctness of batch normalization operations
 * including forward and backward passes in both training and inference modes.
 */
class LegacyBatchNormLayerTest : public ::testing::Test {
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

  void verify_output_shape(const ConstTensor &input, const ConstTensor &output) {
    auto input_shape = input->shape();
    auto output_shape = output->shape();
    EXPECT_EQ(output_shape[0], input_shape[0]);
    EXPECT_EQ(output_shape[1], input_shape[1]);
    EXPECT_EQ(output_shape[2], input_shape[2]);
    EXPECT_EQ(output_shape[3], input_shape[3]);
  }

  void verify_forward_result(const ConstTensor &input, const ConstTensor &output,
                             const Vec<float> &expected_mean, const Vec<float> &expected_var,
                             float epsilon, const ConstTensor gamma = nullptr,
                             const ConstTensor beta = nullptr, float tolerance = 1e-4f) {
    const float *input_data = input->data_as<float>();
    const float *output_data = output->data_as<float>();
    const float *gamma_data = gamma ? gamma->data_as<float>() : nullptr;
    const float *beta_data = beta ? beta->data_as<float>() : nullptr;

    auto input_shape = input->shape();
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t height = input_shape[2];
    size_t width = input_shape[3];

    for (size_t c = 0; c < channels; ++c) {
      float mean = expected_mean[c];
      float var = expected_var[c];
      float inv_std = 1.0f / std::sqrt(var + epsilon);

      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t idx = ((n * channels + c) * height + h) * width + w;
            float normalized = (input_data[idx] - mean) * inv_std;
            float expected = normalized;

            if (gamma_data && beta_data) {
              expected = normalized * gamma_data[c] + beta_data[c];
            }

            EXPECT_NEAR(output_data[idx], expected, tolerance)
                << "Mismatch at batch=" << n << ", channel=" << c << ", h=" << h << ", w=" << w;
          }
        }
      }
    }
  }

  void compute_batch_statistics(const ConstTensor &input, Vec<float> &means, Vec<float> &vars) {
    const float *data = input->data_as<float>();
    auto input_shape = input->shape();
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t height = input_shape[2];
    size_t width = input_shape[3];
    size_t spatial_size = height * width;
    size_t batch_spatial = batch_size * spatial_size;

    means.resize(channels, 0.0f);
    vars.resize(channels, 0.0f);

    for (size_t c = 0; c < channels; ++c) {
      float sum = 0.0f;
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t idx = ((n * channels + c) * height + h) * width + w;
            sum += data[idx];
          }
        }
      }
      means[c] = sum / batch_spatial;
    }

    for (size_t c = 0; c < channels; ++c) {
      float sum_sq = 0.0f;
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t idx = ((n * channels + c) * height + h) * width + w;
            float diff = data[idx] - means[c];
            sum_sq += diff * diff;
          }
        }
      }
      vars[c] = sum_sq / batch_spatial;
    }
  }

  bool has_cpu_;
};

TEST_F(LegacyBatchNormLayerTest, BasicForwardPassTraining) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(3, 1e-5f, 0.1f, false, "test_bn");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);

  Vec<float> means, vars;
  compute_batch_statistics(input, means, vars);

  verify_forward_result(input, output, means, vars, 1e-5f);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassWithAffineTraining) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(3, 1e-5f, 0.1f, true, "test_bn_affine");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);

  auto params = node.parameters();
  EXPECT_EQ(params.size(), 4);

  Vec<float> means, vars;
  compute_batch_statistics(input, means, vars);

  verify_forward_result(input, output, means, vars, 1e-5f, params[0], params[1]);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassSingleChannel) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(1, 1e-5f, 0.1f, false, "test_bn_single");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({4, 1, 8, 8}, getHost());
  input->fill(2.5f);

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);

  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-3f);
  }
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassMultiBatch) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, false, "test_bn_multibatch");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({8, 2, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 20) - 10);
  }

  Tensor output = node.forward({input})[0];

  auto output_shape = output->shape();

  verify_output_shape(input, output);
  EXPECT_EQ(output_shape[0], 8);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassLargeFeatures) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(64, 1e-5f, 0.1f, true, "test_bn_large");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 64, 8, 8}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 100) / 10.0f;
  }

  Tensor output = node.forward({input})[0];

  auto output_shape = output->shape();

  verify_output_shape(input, output);
  EXPECT_EQ(output_shape[1], 64);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassInference) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(3, 1e-5f, 0.1f, false, "test_bn_inference");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(false);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);

  const float *output_data = output->data_as<float>();
  float expected_scale = 1.0f / std::sqrt(1.0f + 1e-5f);
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], input_data[i] * expected_scale, 1e-3f);
  }
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassInferenceWithAffine) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, true, "test_bn_inference_affine");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(false);

  Tensor input = make_tensor<float>({1, 2, 4, 4}, getHost());
  input->fill(1.0f);

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, BasicBackwardPass) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, false, "test_bn_backward");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = node.forward({input})[0];

  Tensor grad_output = make_tensor<float>(output->shape(), getHost());
  grad_output->fill(1.0f);

  Tensor grad_input = node.backward({grad_output})[0];

  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyBatchNormLayerTest, BackwardPassWithAffine) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(3, 1e-5f, 0.1f, true, "test_bn_backward_affine");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = node.forward({input})[0];

  Tensor grad_output = make_tensor<float>(output->shape(), getHost());
  float *grad_data = grad_output->data_as<float>();
  for (size_t i = 0; i < grad_output->size(); ++i) {
    grad_data[i] = static_cast<float>(i % 5) / 5.0f;
  }

  Tensor grad_input = node.backward({grad_output})[0];

  EXPECT_EQ(grad_input->shape(), input->shape());

  auto grads = node.gradients();
  EXPECT_EQ(grads.size(), 4);
}

TEST_F(LegacyBatchNormLayerTest, BackwardPassMultiBatch) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, false, "test_bn_backward_multibatch");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({8, 2, 4, 4}, getHost());
  input->fill(1.0f);

  Tensor output = node.forward({input})[0];

  Tensor grad_output = make_tensor<float>(output->shape(), getHost());
  grad_output->fill(1.0f);

  Tensor grad_input = node.backward({grad_output})[0];

  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[0], 8);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyBatchNormLayerTest, BackwardPassZeroGradient) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, true, "test_bn_backward_zero");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  input->fill(1.0f);

  Tensor output = node.forward({input})[0];

  Tensor grad_output = make_tensor<float>(output->shape(), getHost());
  grad_output->fill(0.0f);

  Tensor grad_input = node.backward({grad_output})[0];

  EXPECT_EQ(grad_input->shape(), input->shape());

  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(LegacyBatchNormLayerTest, ComputeOutputShape) {
  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(16, 1e-5f, 0.1f, true, "test_bn_shape");
  LegacyBatchNormLayer *layer = bn_layer.get();

  Vec<size_t> input_shape = {4, 16, 32, 32};
  Vec<size_t> expected_shape = {4, 16, 32, 32};

  Vec<size_t> output_shape = layer->output_shapes({input_shape})[0];

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyBatchNormLayerTest, GetConfig) {
  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(32, 1e-4f, 0.2f, true, "test_bn_config");
  LegacyBatchNormLayer *layer = bn_layer.get();

  LayerConfig config = layer->get_config();

  EXPECT_EQ(config.name, "test_bn_config");
  EXPECT_EQ(config.get<size_t>("num_features"), 32);
  EXPECT_NEAR(config.get<float>("epsilon"), 1e-4f, 1e-8f);
  EXPECT_NEAR(config.get<float>("momentum"), 0.2f, 1e-8f);
  EXPECT_EQ(config.get<bool>("affine"), true);
}

TEST_F(LegacyBatchNormLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_bn_from_config";
  config.set("num_features", size_t(64));
  config.set("epsilon", 1e-5f);
  config.set("momentum", 0.1f);
  config.set("affine", true);

  auto layer = LegacyBatchNormLayer::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  LayerConfig retrieved_config = layer->get_config();
  EXPECT_EQ(retrieved_config.get<size_t>("num_features"), 64);
}

TEST_F(LegacyBatchNormLayerTest, ParameterCollectionWithAffine) {
  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(16, 1e-5f, 0.1f, true, "test_bn_params_affine");
  LegacyBatchNormLayer *layer = bn_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(bn_layer));
  auto graph = builder.compile(allocator);

  Vec<Tensor> params = layer->parameters();

  EXPECT_EQ(params.size(), 2);
}

TEST_F(LegacyBatchNormLayerTest, ParameterCollectionWithoutAffine) {
  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(16, 1e-5f, 0.1f, false, "test_bn_params_no_affine");
  LegacyBatchNormLayer *layer = bn_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(bn_layer));
  auto graph = builder.compile(allocator);

  Vec<Tensor> params = layer->parameters();

  EXPECT_EQ(params.size(), 2);
}

TEST_F(LegacyBatchNormLayerTest, GradientCollectionWithAffine) {
  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(16, 1e-5f, 0.1f, true, "test_bn_grads_affine");
  LegacyBatchNormLayer *layer = bn_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(bn_layer));
  auto graph = builder.compile(allocator);

  Vec<Tensor> grads = layer->gradients();

  EXPECT_EQ(grads.size(), 4);
}

TEST_F(LegacyBatchNormLayerTest, GradientCollectionWithoutAffine) {
  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(16, 1e-5f, 0.1f, false, "test_bn_grads_no_affine");
  LegacyBatchNormLayer *layer = bn_layer.get();
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;
  builder.add_layer(std::move(bn_layer));
  auto graph = builder.compile(allocator);

  Vec<Tensor> grads = layer->gradients();

  EXPECT_EQ(grads.size(), 2);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseSmallBatch) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(3, 1e-5f, 0.1f, false, "test_bn_small_batch");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({1, 3, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseLargeEpsilon) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-1f, 0.1f, false, "test_bn_large_epsilon");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  input->fill(1.0f);

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseSmallSpatialSize) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(4, 1e-5f, 0.1f, true, "test_bn_small_spatial");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({4, 4, 1, 1}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor output = node.forward({input})[0];

  auto output_shape = output->shape();

  verify_output_shape(input, output);
  EXPECT_EQ(output_shape[2], 1);
  EXPECT_EQ(output_shape[3], 1);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseLargeValues) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, false, "test_bn_large_values");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  input->fill(1e6f);

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);

  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-3f);
  }
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseNegativeValues) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, true, "test_bn_negative");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, NumericalStabilitySmallValues) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer =
      std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, false, "test_bn_small_values");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  input->fill(1e-6f);

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, NumericalStabilityMixedValues) {
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  GraphBuilder builder;

  auto bn_layer = std::make_unique<LegacyBatchNormLayer>(2, 1e-5f, 0.1f, true, "test_bn_mixed");
  auto &node = builder.add_layer(std::move(bn_layer));

  Graph graph = builder.compile(allocator);
  node.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getHost());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
  }

  Tensor output = node.forward({input})[0];

  verify_output_shape(input, output);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
