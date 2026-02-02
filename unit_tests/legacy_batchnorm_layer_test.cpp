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
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

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

  void verify_output_shape(const Tensor &input, const Tensor &output) {
    auto input_shape = input->shape();
    auto output_shape = output->shape();
    EXPECT_EQ(output_shape[0], input_shape[0]);
    EXPECT_EQ(output_shape[1], input_shape[1]);
    EXPECT_EQ(output_shape[2], input_shape[2]);
    EXPECT_EQ(output_shape[3], input_shape[3]);
  }

  void verify_forward_result(const Tensor &input, const Tensor &output,
                             const std::vector<float> &expected_mean,
                             const std::vector<float> &expected_var, float epsilon,
                             const Tensor gamma = nullptr, const Tensor beta = nullptr,
                             float tolerance = 1e-4f) {
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

  void compute_batch_statistics(const Tensor &input, std::vector<float> &means,
                                std::vector<float> &vars) {
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
  LegacyBatchNormLayer layer(3, 1e-5f, 0.1f, false, "test_bn");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);

  std::vector<float> means, vars;
  compute_batch_statistics(input, means, vars);

  verify_forward_result(input, output, means, vars, 1e-5f);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassWithAffineTraining) {
  LegacyBatchNormLayer layer(3, 1e-5f, 0.1f, true, "test_bn_affine");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);

  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 2);

  std::vector<float> means, vars;
  compute_batch_statistics(input, means, vars);

  verify_forward_result(input, output, means, vars, 1e-5f, params[0], params[1]);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassSingleChannel) {
  LegacyBatchNormLayer layer(1, 1e-5f, 0.1f, false, "test_bn_single");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({4, 1, 8, 8}, getCPU());
  input->fill(2.5f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);

  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-3f);
  }
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassMultiBatch) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, false, "test_bn_multibatch");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({8, 2, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 20) - 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
  EXPECT_EQ(output_shape[0], 8);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassLargeFeatures) {
  LegacyBatchNormLayer layer(64, 1e-5f, 0.1f, true, "test_bn_large");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 64, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 100) / 10.0f;
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
  EXPECT_EQ(output_shape[1], 64);
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassInference) {
  LegacyBatchNormLayer layer(3, 1e-5f, 0.1f, false, "test_bn_inference");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(false);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);

  const float *output_data = output->data_as<float>();
  float expected_scale = 1.0f / std::sqrt(1.0f + 1e-5f);
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], input_data[i] * expected_scale, 1e-3f);
  }
}

TEST_F(LegacyBatchNormLayerTest, ForwardPassInferenceWithAffine) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, true, "test_bn_inference_affine");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(false);

  Tensor input = make_tensor<float>({1, 2, 4, 4}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, BasicBackwardPass) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, false, "test_bn_backward");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyBatchNormLayerTest, BackwardPassWithAffine) {
  LegacyBatchNormLayer layer(3, 1e-5f, 0.1f, true, "test_bn_backward_affine");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 3, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  float *grad_data = gradient->data_as<float>();
  for (size_t i = 0; i < gradient->size(); ++i) {
    grad_data[i] = static_cast<float>(i % 5) / 5.0f;
  }

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());

  auto grads = layer.gradients();
  EXPECT_EQ(grads.size(), 2);
}

TEST_F(LegacyBatchNormLayerTest, BackwardPassMultiBatch) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, false, "test_bn_backward_multibatch");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({8, 2, 4, 4}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[0], 8);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyBatchNormLayerTest, BackwardPassZeroGradient) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, true, "test_bn_backward_zero");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(0.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());

  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(LegacyBatchNormLayerTest, ComputeOutputShape) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, true, "test_bn_shape");

  std::vector<size_t> input_shape = {4, 16, 32, 32};
  std::vector<size_t> expected_shape = {4, 16, 32, 32};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyBatchNormLayerTest, GetConfig) {
  LegacyBatchNormLayer layer(32, 1e-4f, 0.2f, true, "test_bn_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_bn_config");
  EXPECT_EQ(config.get<size_t>("num_features"), 32);
  EXPECT_NEAR(config.get<float>("epsilon"), 1e-4f, 1e-8f);
  EXPECT_NEAR(config.get<float>("momentum"), 0.2f, 1e-8f);
  EXPECT_EQ(config.get<bool>("affine"), true);
}

TEST_F(LegacyBatchNormLayerTest, Clone) {
  LegacyBatchNormLayer original(16, 1e-5f, 0.1f, true, "test_bn_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "legacy_batchnorm");
  EXPECT_EQ(cloned->type(), original.type());
}

TEST_F(LegacyBatchNormLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_bn_from_config";
  config.parameters["num_features"] = size_t(64);
  config.parameters["epsilon"] = 1e-5f;
  config.parameters["momentum"] = 0.1f;
  config.parameters["affine"] = true;

  auto layer = LegacyBatchNormLayer::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  LayerConfig retrieved_config = layer->get_config();
  EXPECT_EQ(retrieved_config.get<size_t>("num_features"), 64);
}

TEST_F(LegacyBatchNormLayerTest, ParameterCollectionWithAffine) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, true, "test_bn_params_affine");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> params = layer.parameters();

  EXPECT_EQ(params.size(), 2);
}

TEST_F(LegacyBatchNormLayerTest, ParameterCollectionWithoutAffine) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, false, "test_bn_params_no_affine");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> params = layer.parameters();

  EXPECT_EQ(params.size(), 0);
}

TEST_F(LegacyBatchNormLayerTest, GradientCollectionWithAffine) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, true, "test_bn_grads_affine");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 2);
}

TEST_F(LegacyBatchNormLayerTest, GradientCollectionWithoutAffine) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, false, "test_bn_grads_no_affine");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 0);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseSmallBatch) {
  LegacyBatchNormLayer layer(3, 1e-5f, 0.1f, false, "test_bn_small_batch");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({1, 3, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseLargeEpsilon) {
  LegacyBatchNormLayer layer(2, 1e-1f, 0.1f, false, "test_bn_large_epsilon");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseSmallSpatialSize) {
  LegacyBatchNormLayer layer(4, 1e-5f, 0.1f, true, "test_bn_small_spatial");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({4, 4, 1, 1}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
  EXPECT_EQ(output_shape[2], 1);
  EXPECT_EQ(output_shape[3], 1);
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseLargeValues) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, false, "test_bn_large_values");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  input->fill(1e6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);

  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 0.0f, 1e-3f);
  }
}

TEST_F(LegacyBatchNormLayerTest, EdgeCaseNegativeValues) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, true, "test_bn_negative");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, NumericalStabilitySmallValues) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, false, "test_bn_small_values");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  input->fill(1e-6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, NumericalStabilityMixedValues) {
  LegacyBatchNormLayer layer(2, 1e-5f, 0.1f, true, "test_bn_mixed");
  layer.set_device(getCPU());
  layer.init();
  layer.set_training(true);

  Tensor input = make_tensor<float>({2, 2, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output);
}

TEST_F(LegacyBatchNormLayerTest, ForwardFlopsComputation) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, true, "test_bn_flops");

  std::vector<size_t> input_shape = {4, 16, 32, 32};
  uint64_t flops = layer.forward_flops(input_shape);

  EXPECT_GT(flops, 0);
}

TEST_F(LegacyBatchNormLayerTest, BackwardFlopsComputation) {
  LegacyBatchNormLayer layer(16, 1e-5f, 0.1f, true, "test_bn_backward_flops");

  std::vector<size_t> input_shape = {4, 16, 32, 32};
  uint64_t flops = layer.backward_flops(input_shape);

  EXPECT_GT(flops, 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
