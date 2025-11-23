/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers_impl/batchnorm_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

/**
 * Test fixture for BatchNormLayer validation tests.
 * These tests verify the mathematical correctness of batch normalization operations
 * including forward and backward passes in both training and inference modes.
 */
class BatchNormLayerTest : public ::testing::Test {
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

  // Verify output shape matches input shape
  void verify_output_shape(const Tensor<float> &input, const Tensor<float> &output) {
    EXPECT_EQ(output.batch_size(), input.batch_size());
    EXPECT_EQ(output.channels(), input.channels());
    EXPECT_EQ(output.height(), input.height());
    EXPECT_EQ(output.width(), input.width());
  }

  // Verify forward pass numerical correctness
  void verify_forward_result(const Tensor<float> &input, const Tensor<float> &output,
                             const std::vector<float> &expected_mean,
                             const std::vector<float> &expected_var, float epsilon,
                             const Tensor<float> *gamma = nullptr,
                             const Tensor<float> *beta = nullptr, float tolerance = 1e-4f) {
    const float *input_data = input.data();
    const float *output_data = output.data();
    const float *gamma_data = gamma ? gamma->data() : nullptr;
    const float *beta_data = beta ? beta->data() : nullptr;

    size_t batch_size = input.batch_size();
    size_t channels = input.channels();
    size_t height = input.height();
    size_t width = input.width();
    size_t spatial_size = height * width;

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

  // Compute batch statistics for verification
  void compute_batch_statistics(const Tensor<float> &input, std::vector<float> &means,
                                std::vector<float> &vars) {
    const float *data = input.data();
    size_t batch_size = input.batch_size();
    size_t channels = input.channels();
    size_t height = input.height();
    size_t width = input.width();
    size_t spatial_size = height * width;
    size_t batch_spatial = batch_size * spatial_size;

    means.resize(channels, 0.0f);
    vars.resize(channels, 0.0f);

    // Compute means
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

    // Compute variances
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
  const Device *cpu_device_;
};

// Forward Pass Tests - Training Mode

TEST_F(BatchNormLayerTest, BasicForwardPassTraining) {
  BatchNormLayer<float> layer(3, 1e-5f, 0.1f, false, "test_bn");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 3, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);

  std::vector<float> means, vars;
  compute_batch_statistics(input, means, vars);

  verify_forward_result(input, output, means, vars, 1e-5f);
}

TEST_F(BatchNormLayerTest, ForwardPassWithAffineTraining) {
  BatchNormLayer<float> layer(3, 1e-5f, 0.1f, true, "test_bn_affine");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 3, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);

  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 4); // gamma, beta, running_mean, running_var

  std::vector<float> means, vars;
  compute_batch_statistics(input, means, vars);

  verify_forward_result(input, output, means, vars, 1e-5f, params[0], params[1]);
}

TEST_F(BatchNormLayerTest, ForwardPassSingleChannel) {
  BatchNormLayer<float> layer(1, 1e-5f, 0.1f, false, "test_bn_single");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({4, 1, 8, 8}, cpu_device_);
  input.fill(2.5f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);

  // With constant input, variance should be near zero, output should be near zero
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-3f);
  }
}

TEST_F(BatchNormLayerTest, ForwardPassMultiBatch) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({8, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 20) - 10);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
  EXPECT_EQ(output.batch_size(), 8);
}

TEST_F(BatchNormLayerTest, ForwardPassLargeFeatures) {
  BatchNormLayer<float> layer(64, 1e-5f, 0.1f, true, "test_bn_large");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 64, 8, 8}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100) / 10.0f;
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
  EXPECT_EQ(output.channels(), 64);
}

// Forward Pass Tests - Inference Mode

TEST_F(BatchNormLayerTest, ForwardPassInference) {
  BatchNormLayer<float> layer(3, 1e-5f, 0.1f, false, "test_bn_inference");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(false);

  Tensor<float> input({2, 3, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);

  // In inference mode with initialized running stats (mean=0, var=1),
  // output should be approximately (input - 0) / sqrt(1 + eps)
  const float *output_data = output.data();
  float expected_scale = 1.0f / std::sqrt(1.0f + 1e-5f);
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], input_data[i] * expected_scale, 1e-3f);
  }
}

TEST_F(BatchNormLayerTest, ForwardPassInferenceWithAffine) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, true, "test_bn_inference_affine");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(false);

  Tensor<float> input({1, 2, 4, 4}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
}

// Backward Pass Tests

TEST_F(BatchNormLayerTest, BasicBackwardPass) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_backward");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(BatchNormLayerTest, BackwardPassWithAffine) {
  BatchNormLayer<float> layer(3, 1e-5f, 0.1f, true, "test_bn_backward_affine");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 3, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  float *grad_data = gradient.data();
  for (size_t i = 0; i < gradient.size(); ++i) {
    grad_data[i] = static_cast<float>(i % 5) / 5.0f;
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  EXPECT_EQ(grad_input.shape(), input.shape());

  auto grads = layer.gradients();
  EXPECT_EQ(grads.size(), 4); // gamma_grad, beta_grad, running_mean_grad, running_var_grad
}

TEST_F(BatchNormLayerTest, BackwardPassMultiBatch) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_backward_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({8, 2, 4, 4}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  EXPECT_EQ(grad_input.batch_size(), 8);
  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(BatchNormLayerTest, BackwardPassZeroGradient) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, true, "test_bn_backward_zero");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(0.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  EXPECT_EQ(grad_input.shape(), input.shape());

  // Zero gradient should produce zero input gradient
  const float *grad_input_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.0f, 1e-5f);
  }
}

// Configuration Tests

TEST_F(BatchNormLayerTest, ComputeOutputShape) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, true, "test_bn_shape");

  std::vector<size_t> input_shape = {4, 16, 32, 32};
  std::vector<size_t> expected_shape = {4, 16, 32, 32};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(BatchNormLayerTest, GetConfig) {
  BatchNormLayer<float> layer(32, 1e-4f, 0.2f, true, "test_bn_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_bn_config");
  EXPECT_EQ(config.get<size_t>("num_features"), 32);
  EXPECT_NEAR(config.get<float>("epsilon"), 1e-4f, 1e-8f);
  EXPECT_NEAR(config.get<float>("momentum"), 0.2f, 1e-8f);
  EXPECT_EQ(config.get<bool>("affine"), true);
}

TEST_F(BatchNormLayerTest, Clone) {
  BatchNormLayer<float> original(16, 1e-5f, 0.1f, true, "test_bn_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "batchnorm");
  EXPECT_EQ(cloned->type(), original.type());
}

TEST_F(BatchNormLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_bn_from_config";
  config.parameters["num_features"] = size_t(64);
  config.parameters["epsilon"] = 1e-5f;
  config.parameters["momentum"] = 0.1f;
  config.parameters["affine"] = true;

  auto layer = BatchNormLayer<float>::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  LayerConfig retrieved_config = layer->get_config();
  EXPECT_EQ(retrieved_config.get<size_t>("num_features"), 64);
}

// Parameter and Gradient Tests

TEST_F(BatchNormLayerTest, ParameterCollectionWithAffine) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, true, "test_bn_params_affine");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> params = layer.parameters();

  EXPECT_EQ(params.size(), 4); // gamma, beta, running_mean, running_var
}

TEST_F(BatchNormLayerTest, ParameterCollectionWithoutAffine) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, false, "test_bn_params_no_affine");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> params = layer.parameters();

  EXPECT_EQ(params.size(), 2); // running_mean, running_var only
}

TEST_F(BatchNormLayerTest, GradientCollectionWithAffine) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, true, "test_bn_grads_affine");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 4); // gamma_grad, beta_grad, running_mean_grad, running_var_grad
}

TEST_F(BatchNormLayerTest, GradientCollectionWithoutAffine) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, false, "test_bn_grads_no_affine");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 2); // running_mean_grad, running_var_grad only
}

// Edge Cases

TEST_F(BatchNormLayerTest, EdgeCaseSmallBatch) {
  BatchNormLayer<float> layer(3, 1e-5f, 0.1f, false, "test_bn_small_batch");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({1, 3, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
}

TEST_F(BatchNormLayerTest, EdgeCaseLargeEpsilon) {
  BatchNormLayer<float> layer(2, 1e-1f, 0.1f, false, "test_bn_large_epsilon");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
}

TEST_F(BatchNormLayerTest, EdgeCaseSmallSpatialSize) {
  BatchNormLayer<float> layer(4, 1e-5f, 0.1f, true, "test_bn_small_spatial");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({4, 4, 1, 1}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
  EXPECT_EQ(output.height(), 1);
  EXPECT_EQ(output.width(), 1);
}

TEST_F(BatchNormLayerTest, EdgeCaseLargeValues) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_large_values");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  input.fill(1e6f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);

  // With constant large input, output should normalize to near zero
  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-3f);
  }
}

TEST_F(BatchNormLayerTest, EdgeCaseNegativeValues) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, true, "test_bn_negative");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
}

// Numerical Stability Tests

TEST_F(BatchNormLayerTest, NumericalStabilitySmallValues) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_small_values");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  input.fill(1e-6f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
}

TEST_F(BatchNormLayerTest, NumericalStabilityMixedValues) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, true, "test_bn_mixed");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output);
}

// Running Statistics Tests

TEST_F(BatchNormLayerTest, RunningStatisticsUpdate) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_running_stats");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  auto params_before = layer.parameters();
  Tensor<float> running_mean_before = params_before[0]->clone();
  Tensor<float> running_var_before = params_before[1]->clone();

  Tensor<float> input({2, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  layer.forward(input);

  auto params_after = layer.parameters();
  const float *mean_before = running_mean_before.data();
  const float *mean_after = params_after[0]->data();

  // Running statistics should have changed (unless batch stats happen to match initial values)
  // This is a weak test but checks that the mechanism is working
  bool stats_updated = false;
  for (size_t i = 0; i < params_after[0]->size(); ++i) {
    if (std::abs(mean_after[i] - mean_before[i]) > 1e-6f) {
      stats_updated = true;
      break;
    }
  }
  // With the input pattern, stats should update
  EXPECT_TRUE(stats_updated);
}

TEST_F(BatchNormLayerTest, InferenceUsesRunningStats) {
  BatchNormLayer<float> layer(2, 1e-5f, 0.1f, false, "test_bn_inference_stats");
  layer.set_device(cpu_device_);
  layer.initialize();

  // First, do a training forward pass to update running stats
  layer.set_training(true);
  Tensor<float> train_input({4, 2, 4, 4}, cpu_device_);
  float *train_data = train_input.data();
  for (size_t i = 0; i < train_input.size(); ++i) {
    train_data[i] = static_cast<float>(i % 10);
  }
  layer.forward(train_input);

  // Now switch to inference mode
  layer.set_training(false);
  Tensor<float> test_input({1, 2, 4, 4}, cpu_device_);
  test_input.fill(5.0f);

  const Tensor<float> &output = layer.forward(test_input);

  verify_output_shape(test_input, output);
  // Output should be based on running stats, not batch stats
}

// FLOPS Tests

TEST_F(BatchNormLayerTest, ForwardFlopsComputation) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, true, "test_bn_flops");

  std::vector<size_t> input_shape = {4, 16, 32, 32};
  uint64_t flops = layer.forward_flops(input_shape);

  EXPECT_GT(flops, 0);
}

TEST_F(BatchNormLayerTest, BackwardFlopsComputation) {
  BatchNormLayer<float> layer(16, 1e-5f, 0.1f, true, "test_bn_backward_flops");

  std::vector<size_t> input_shape = {4, 16, 32, 32};
  uint64_t flops = layer.backward_flops(input_shape);

  EXPECT_GT(flops, 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
