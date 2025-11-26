/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers_impl/groupnorm_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

/**
 * Test fixture for GroupNormLayer validation tests.
 * These tests verify the mathematical correctness of group normalization operations
 * including forward and backward passes.
 */
class GroupNormLayerTest : public ::testing::Test {
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

  // Verify output shape matches input shape
  void verify_output_shape(const Tensor<float> &input, const Tensor<float> &output) {
    EXPECT_EQ(output.batch_size(), input.batch_size());
    EXPECT_EQ(output.channels(), input.channels());
    EXPECT_EQ(output.height(), input.height());
    EXPECT_EQ(output.width(), input.width());
  }

  // Compute group statistics for verification
  void compute_group_statistics(const Tensor<float> &input, size_t num_groups,
                                std::vector<float> &means, std::vector<float> &vars) {
    const float *data = input.data();
    size_t batch_size = input.batch_size();
    size_t channels = input.channels();
    size_t height = input.height();
    size_t width = input.width();
    size_t spatial_size = height * width;
    size_t channels_per_group = channels / num_groups;
    size_t group_size = channels_per_group * spatial_size;

    means.resize(batch_size * num_groups, 0.0f);
    vars.resize(batch_size * num_groups, 0.0f);

    // Compute means per group
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t g = 0; g < num_groups; ++g) {
        float sum = 0.0f;
        for (size_t c = 0; c < channels_per_group; ++c) {
          size_t global_c = g * channels_per_group + c;
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
              size_t idx = ((n * channels + global_c) * height + h) * width + w;
              sum += data[idx];
            }
          }
        }
        size_t group_idx = n * num_groups + g;
        means[group_idx] = sum / group_size;
      }
    }

    // Compute variances per group
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t g = 0; g < num_groups; ++g) {
        size_t group_idx = n * num_groups + g;
        float sum_sq = 0.0f;
        for (size_t c = 0; c < channels_per_group; ++c) {
          size_t global_c = g * channels_per_group + c;
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
              size_t idx = ((n * channels + global_c) * height + h) * width + w;
              float diff = data[idx] - means[group_idx];
              sum_sq += diff * diff;
            }
          }
        }
        vars[group_idx] = sum_sq / group_size;
      }
    }
  }

  // Verify forward pass numerical correctness
  void verify_forward_result(const Tensor<float> &input, const Tensor<float> &output,
                             size_t num_groups, const std::vector<float> &expected_mean,
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
    size_t channels_per_group = channels / num_groups;

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        size_t g = c / channels_per_group;
        size_t group_idx = n * num_groups + g;
        float mean = expected_mean[group_idx];
        float var = expected_var[group_idx];
        float inv_std = 1.0f / std::sqrt(var + epsilon);

        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t idx = ((n * channels + c) * height + h) * width + w;
            float normalized = (input_data[idx] - mean) * inv_std;
            float expected = normalized;

            if (gamma_data && beta_data) {
              expected = normalized * gamma_data[c] + beta_data[c];
            }

            EXPECT_NEAR(output_data[idx], expected, tolerance)
                << "Mismatch at batch=" << n << ", channel=" << c << ", h=" << h << ", w=" << w
                << ", group=" << g;
          }
        }
      }
    }
  }

  bool has_cpu_;
  const Device *cpu_device_;
};

// Forward Pass Tests

TEST_F(GroupNormLayerTest, BasicForwardPass) {
  size_t num_groups = 2;
  size_t num_channels = 4;
  GroupNormLayer<float> layer(num_groups, num_channels, 1e-5f, false, "test_gn");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, num_channels, 3, 3}, cpu_device_);

  // Initialize with simple values
  float *data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);
  verify_output_shape(input, output);

  std::vector<float> expected_mean, expected_var;
  compute_group_statistics(input, num_groups, expected_mean, expected_var);
  verify_forward_result(input, output, num_groups, expected_mean, expected_var, 1e-5f);
}

TEST_F(GroupNormLayerTest, ForwardPassWithAffine) {
  size_t num_groups = 2;
  size_t num_channels = 4;
  GroupNormLayer<float> layer(num_groups, num_channels, 1e-5f, true, "test_gn_affine");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, num_channels, 3, 3}, cpu_device_);

  float *data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    data[i] = static_cast<float>(i % 10) + 1.0f;
  }

  const Tensor<float> &output = layer.forward(input);
  verify_output_shape(input, output);

  std::vector<float> expected_mean, expected_var;
  compute_group_statistics(input, num_groups, expected_mean, expected_var);

  // Get gamma and beta for verification
  std::vector<Tensor<float> *> params = layer.parameters();
  ASSERT_EQ(params.size(), 2);

  verify_forward_result(input, output, num_groups, expected_mean, expected_var, 1e-5f, params[0],
                        params[1]);
}

TEST_F(GroupNormLayerTest, SingleGroup) {
  // Single group is equivalent to LayerNorm
  size_t num_groups = 1;
  size_t num_channels = 4;
  GroupNormLayer<float> layer(num_groups, num_channels, 1e-5f, false, "test_gn_single");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, num_channels, 2, 2}, cpu_device_);

  float *data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    data[i] = static_cast<float>(i) + 1.0f;
  }

  const Tensor<float> &output = layer.forward(input);
  verify_output_shape(input, output);

  std::vector<float> expected_mean, expected_var;
  compute_group_statistics(input, num_groups, expected_mean, expected_var);
  verify_forward_result(input, output, num_groups, expected_mean, expected_var, 1e-5f);
}

TEST_F(GroupNormLayerTest, ChannelsEqualsGroups) {
  // num_groups == num_channels is equivalent to InstanceNorm
  size_t num_groups = 4;
  size_t num_channels = 4;
  GroupNormLayer<float> layer(num_groups, num_channels, 1e-5f, false, "test_gn_instance");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, num_channels, 3, 3}, cpu_device_);

  float *data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    data[i] = static_cast<float>((i * 3) % 7) + 0.5f;
  }

  const Tensor<float> &output = layer.forward(input);
  verify_output_shape(input, output);

  std::vector<float> expected_mean, expected_var;
  compute_group_statistics(input, num_groups, expected_mean, expected_var);
  verify_forward_result(input, output, num_groups, expected_mean, expected_var, 1e-5f);
}

TEST_F(GroupNormLayerTest, BackwardPassGradientFlow) {
  size_t num_groups = 2;
  size_t num_channels = 4;
  GroupNormLayer<float> layer(num_groups, num_channels, 1e-5f, true, "test_gn_backward");
  layer.set_device(cpu_device_);
  layer.initialize();
  layer.set_training(true);

  Tensor<float> input({2, num_channels, 3, 3}, cpu_device_);

  float *data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    data[i] = static_cast<float>(i % 10) + 1.0f;
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> grad_output = output.clone();
  grad_output.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(grad_output);

  EXPECT_EQ(grad_input.batch_size(), input.batch_size());
  EXPECT_EQ(grad_input.channels(), input.channels());
  EXPECT_EQ(grad_input.height(), input.height());
  EXPECT_EQ(grad_input.width(), input.width());

  // Verify gradients are non-zero
  const float *grad_data = grad_input.data();
  bool has_nonzero = false;
  for (size_t i = 0; i < grad_input.size(); ++i) {
    if (std::abs(grad_data[i]) > 1e-6f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Gradient should contain non-zero values";
}

TEST_F(GroupNormLayerTest, InvalidConfiguration) {
  // num_channels not divisible by num_groups should throw
  EXPECT_THROW(
      { GroupNormLayer<float> layer(3, 5, 1e-5f, true, "invalid"); }, std::invalid_argument);
}

TEST_F(GroupNormLayerTest, ConfigurationRoundTrip) {
  size_t num_groups = 2;
  size_t num_channels = 6;
  GroupNormLayer<float> original(num_groups, num_channels, 1e-5f, true, "test_config");

  LayerConfig config = original.get_config();
  EXPECT_EQ(config.name, "test_config");
  EXPECT_EQ(config.get<size_t>("num_groups"), num_groups);
  EXPECT_EQ(config.get<size_t>("num_channels"), num_channels);
  EXPECT_FLOAT_EQ(config.get<float>("epsilon"), 1e-5f);
  EXPECT_TRUE(config.get<bool>("affine"));

  auto restored = GroupNormLayer<float>::create_from_config(config);
  EXPECT_NE(restored, nullptr);
  EXPECT_EQ(restored->type(), "groupnorm");
}

TEST_F(GroupNormLayerTest, CloneLayer) {
  size_t num_groups = 2;
  size_t num_channels = 4;
  GroupNormLayer<float> original(num_groups, num_channels, 1e-5f, true, "test_clone");

  auto cloned = original.clone();
  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "groupnorm");

  LayerConfig original_config = original.get_config();
  LayerConfig cloned_config = cloned->get_config();

  EXPECT_EQ(cloned_config.get<size_t>("num_groups"), original_config.get<size_t>("num_groups"));
  EXPECT_EQ(cloned_config.get<size_t>("num_channels"), original_config.get<size_t>("num_channels"));
  EXPECT_FLOAT_EQ(cloned_config.get<float>("epsilon"), original_config.get<float>("epsilon"));
  EXPECT_EQ(cloned_config.get<bool>("affine"), original_config.get<bool>("affine"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
