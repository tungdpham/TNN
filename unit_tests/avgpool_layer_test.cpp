/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers_impl/avgpool2d_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

/**
 * Test fixture for AvgPool2DLayer validation tests.
 * These tests verify the mathematical correctness of average pooling operations
 * including forward and backward passes.
 */
class AvgPool2DLayerTest : public ::testing::Test {
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

  // Verify forward pass mathematically
  void verify_forward_result(const Tensor<float> &input, const Tensor<float> &output, size_t pool_h,
                             size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                             size_t pad_w, float tolerance = 1e-5f) {
    const float *input_data = input.data();
    const float *output_data = output.data();

    size_t batch_size = input.batch_size();
    size_t channels = input.channels();
    size_t input_h = input.height();
    size_t input_w = input.width();

    size_t padded_h = input_h + 2 * pad_h;
    size_t padded_w = input_w + 2 * pad_w;
    size_t output_h = (padded_h - pool_h) / stride_h + 1;
    size_t output_w = (padded_w - pool_w) / stride_w + 1;

    EXPECT_EQ(output.batch_size(), batch_size);
    EXPECT_EQ(output.channels(), channels);
    EXPECT_EQ(output.height(), output_h);
    EXPECT_EQ(output.width(), output_w);

    // Verify each output element
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            float expected_sum = 0.0f;

            for (size_t ph = 0; ph < pool_h; ++ph) {
              for (size_t pw = 0; pw < pool_w; ++pw) {
                int h_idx = static_cast<int>(oh * stride_h + ph) - static_cast<int>(pad_h);
                int w_idx = static_cast<int>(ow * stride_w + pw) - static_cast<int>(pad_w);

                if (h_idx >= 0 && h_idx < static_cast<int>(input_h) && w_idx >= 0 &&
                    w_idx < static_cast<int>(input_w)) {
                  size_t input_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
                  expected_sum += input_data[input_idx];
                }
              }
            }
            float expected_avg = expected_sum / (pool_h * pool_w);
            size_t output_idx = ((n * channels + c) * output_h + oh) * output_w + ow;
            float actual = output_data[output_idx];

            EXPECT_NEAR(actual, expected_avg, tolerance)
                << "Mismatch at batch=" << n << ", channel=" << c << ", height=" << oh
                << ", width=" << ow << ". Expected: " << expected_avg << ", Got: " << actual;
          }
        }
      }
    }
  }

  // Verify backward pass mathematically
  void verify_backward_result(const Tensor<float> &gradient, const Tensor<float> &grad_input,
                              size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                              size_t pad_h, size_t pad_w, float tolerance = 1e-5f) {
    const float *grad_data = gradient.data();
    const float *grad_input_data = grad_input.data();

    size_t batch_size = gradient.batch_size();
    size_t channels = gradient.channels();
    size_t output_h = gradient.height();
    size_t output_w = gradient.width();

    size_t input_h = grad_input.height();
    size_t input_w = grad_input.width();

    std::vector<float> expected_grad_input(grad_input.size(), 0.0f);
    float pool_size_inv = 1.0f / (pool_h * pool_w);

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            size_t output_idx = ((n * channels + c) * output_h + oh) * output_w + ow;
            float grad_val = grad_data[output_idx] * pool_size_inv;

            for (size_t ph = 0; ph < pool_h; ++ph) {
              for (size_t pw = 0; pw < pool_w; ++pw) {
                int h_idx = static_cast<int>(oh * stride_h + ph) - static_cast<int>(pad_h);
                int w_idx = static_cast<int>(ow * stride_w + pw) - static_cast<int>(pad_w);

                if (h_idx >= 0 && h_idx < static_cast<int>(input_h) && w_idx >= 0 &&
                    w_idx < static_cast<int>(input_w)) {
                  size_t input_idx = ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
                  expected_grad_input[input_idx] += grad_val;
                }
              }
            }
          }
        }
      }
    }

    for (size_t i = 0; i < grad_input.size(); ++i) {
      EXPECT_NEAR(grad_input_data[i], expected_grad_input[i], tolerance)
          << "Gradient mismatch at index " << i;
    }
  }

  bool has_cpu_;
  const Device *cpu_device_;
};

// Forward Pass Tests

TEST_F(AvgPool2DLayerTest, BasicForwardPass) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_avgpool");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 2, 2, 2, 2, 0, 0);

  const float *output_data = output.data();
  EXPECT_NEAR(output_data[0], 3.5f, 1e-5f);
  EXPECT_NEAR(output_data[1], 5.5f, 1e-5f);
  EXPECT_NEAR(output_data[2], 11.5f, 1e-5f);
  EXPECT_NEAR(output_data[3], 13.5f, 1e-5f);
}

TEST_F(AvgPool2DLayerTest, ForwardPassWithStride) {
  AvgPool2DLayer<float> layer(3, 3, 1, 1, 0, 0, "test_avgpool_stride");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 25; ++i) {
    input_data[i] = 1.0f;
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 3, 3, 1, 1, 0, 0);

  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 1.0f, 1e-5f);
  }
}

TEST_F(AvgPool2DLayerTest, ForwardPassWithPadding) {
  AvgPool2DLayer<float> layer(3, 3, 1, 1, 1, 1, "test_avgpool_padding");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 3, 3}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 9; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  EXPECT_EQ(output.height(), 3);
  EXPECT_EQ(output.width(), 3);

  verify_forward_result(input, output, 3, 3, 1, 1, 1, 1);
}

TEST_F(AvgPool2DLayerTest, ForwardPassMultiChannel) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_avgpool_multichannel");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();

  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 2, 2, 2, 2, 0, 0);

  EXPECT_EQ(output.batch_size(), 1);
  EXPECT_EQ(output.channels(), 2);
  EXPECT_EQ(output.height(), 2);
  EXPECT_EQ(output.width(), 2);
}

TEST_F(AvgPool2DLayerTest, ForwardPassMultiBatch) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_avgpool_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({2, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();

  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 2, 2, 2, 2, 0, 0);

  EXPECT_EQ(output.batch_size(), 2);
  EXPECT_EQ(output.channels(), 1);
  EXPECT_EQ(output.height(), 2);
  EXPECT_EQ(output.width(), 2);
}

TEST_F(AvgPool2DLayerTest, ForwardPassNonSquarePooling) {
  AvgPool2DLayer<float> layer(3, 2, 2, 2, 0, 0, "test_avgpool_nonsquare");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 6, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 24; ++i) {
    input_data[i] = 1.0f;
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 3, 2, 2, 2, 0, 0);

  const float *output_data = output.data();
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output_data[i], 1.0f, 1e-5f);
  }
}

// Backward Pass Tests

TEST_F(AvgPool2DLayerTest, BasicBackwardPass) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_avgpool_backward");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  layer.forward(input);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  float *grad_data = gradient.data();
  for (int i = 0; i < 4; ++i) {
    grad_data[i] = 1.0f;
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_backward_result(gradient, grad_input, 2, 2, 2, 2, 0, 0);

  const float *grad_input_data = grad_input.data();
  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.25f, 1e-5f);
  }
}

TEST_F(AvgPool2DLayerTest, BackwardPassWithPadding) {
  AvgPool2DLayer<float> layer(3, 3, 1, 1, 1, 1, "test_avgpool_backward_pad");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 3, 3}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 9; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  float *grad_data = gradient.data();
  for (size_t i = 0; i < gradient.size(); ++i) {
    grad_data[i] = 1.0f;
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_backward_result(gradient, grad_input, 3, 3, 1, 1, 1, 1);

  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(AvgPool2DLayerTest, BackwardPassMultiChannel) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_avgpool_backward_multichannel");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 2, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  float *grad_data = gradient.data();
  for (size_t i = 0; i < gradient.size(); ++i) {
    grad_data[i] = 1.0f;
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_backward_result(gradient, grad_input, 2, 2, 2, 2, 0, 0);

  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(AvgPool2DLayerTest, BackwardPassVariableGradient) {
  AvgPool2DLayer<float> layer(2, 2, 1, 1, 0, 0, "test_avgpool_backward_var");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 3, 3}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 9; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  float *grad_data = gradient.data();
  for (size_t i = 0; i < gradient.size(); ++i) {
    grad_data[i] = static_cast<float>(i + 1);
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_backward_result(gradient, grad_input, 2, 2, 1, 1, 0, 0);

  EXPECT_EQ(grad_input.shape(), input.shape());
}

// Configuration Tests

TEST_F(AvgPool2DLayerTest, ComputeOutputShape) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_avgpool_shape");

  std::vector<size_t> input_shape = {2, 3, 8, 8};
  std::vector<size_t> expected_shape = {2, 3, 4, 4};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(AvgPool2DLayerTest, ComputeOutputShapeWithPadding) {
  AvgPool2DLayer<float> layer(3, 3, 1, 1, 1, 1, "test_avgpool_shape_pad");

  std::vector<size_t> input_shape = {1, 1, 5, 5};
  std::vector<size_t> expected_shape = {1, 1, 5, 5};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(AvgPool2DLayerTest, GetConfig) {
  AvgPool2DLayer<float> layer(3, 4, 2, 1, 1, 2, "test_avgpool_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_avgpool_config");
  EXPECT_EQ(config.get<size_t>("pool_h"), 3);
  EXPECT_EQ(config.get<size_t>("pool_w"), 4);
  EXPECT_EQ(config.get<size_t>("stride_h"), 2);
  EXPECT_EQ(config.get<size_t>("stride_w"), 1);
  EXPECT_EQ(config.get<size_t>("pad_h"), 1);
  EXPECT_EQ(config.get<size_t>("pad_w"), 2);
}

TEST_F(AvgPool2DLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_avgpool_recreate";
  config.parameters["pool_h"] = size_t(2);
  config.parameters["pool_w"] = size_t(2);
  config.parameters["stride_h"] = size_t(2);
  config.parameters["stride_w"] = size_t(2);
  config.parameters["pad_h"] = size_t(0);
  config.parameters["pad_w"] = size_t(0);

  auto layer = AvgPool2DLayer<float>::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  EXPECT_EQ(layer->type(), "avgpool2d");
}

TEST_F(AvgPool2DLayerTest, Clone) {
  AvgPool2DLayer<float> original(3, 3, 1, 1, 1, 1, "test_avgpool_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "avgpool2d");
  EXPECT_EQ(cloned->type(), original.type());
}

// Edge Cases

TEST_F(AvgPool2DLayerTest, EdgeCaseGlobalAveragePooling) {
  AvgPool2DLayer<float> layer(4, 4, 1, 1, 0, 0, "test_global_avgpool");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 2.0f;
  }

  const Tensor<float> &output_ref = layer.forward(input);

  EXPECT_EQ(output_ref.height(), 1);
  EXPECT_EQ(output_ref.width(), 1);
  EXPECT_NEAR(output_ref.data()[0], 2.0f, 1e-5f);
}

TEST_F(AvgPool2DLayerTest, EdgeCaseZeroGradient) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_zero_gradient");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1.0f;
  }

  layer.forward(input);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  gradient.fill(0.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_backward_result(gradient, grad_input, 2, 2, 2, 2, 0, 0);

  for (size_t i = 0; i < grad_input.size(); ++i) {
    EXPECT_NEAR(grad_input.data()[i], 0.0f, 1e-5f);
  }
}

TEST_F(AvgPool2DLayerTest, EdgeCaseLargeValues) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_large_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1e6f;
  }

  const Tensor<float> &output_ref = layer.forward(input);

  verify_forward_result(input, output_ref, 2, 2, 2, 2, 0, 0);

  for (size_t i = 0; i < output_ref.size(); ++i) {
    EXPECT_NEAR(output_ref.data()[i], 1e6f, 1e1f);
  }
}

TEST_F(AvgPool2DLayerTest, EdgeCaseNegativeValues) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_negative_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 2, 2, 2, 2, 0, 0);

  EXPECT_NEAR(output.data()[0], -3.5f, 1e-5f);
}

// Numerical Stability Tests

TEST_F(AvgPool2DLayerTest, NumericalStabilitySmallValues) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_small_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1e-6f;
  }

  const Tensor<float> &output = layer.forward(input);

  verify_forward_result(input, output, 2, 2, 2, 2, 0, 0);

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output.data()[i], 1e-6f, 1e-12f);
  }
}

TEST_F(AvgPool2DLayerTest, BackwardNumericalStability) {
  AvgPool2DLayer<float> layer(2, 2, 2, 2, 0, 0, "test_backward_stability");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 4, 4}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1e-6f;
  }

  layer.forward(input);

  Tensor<float> gradient({1, 1, 2, 2}, cpu_device_);
  float *grad_data = gradient.data();
  for (size_t i = 0; i < gradient.size(); ++i) {
    grad_data[i] = 1e-6f;
  }

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_backward_result(gradient, grad_input, 2, 2, 2, 2, 0, 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
