/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/legacy_conv2d_layer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "device/device_manager.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Test fixture for LegacyConv2DLayer validation tests.
 * These tests verify the mathematical correctness of 2D convolution operations
 * including forward and backward passes.
 */
class LegacyConv2DLayerTest : public ::testing::Test {
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

  void verify_output_shape(const Tensor &input, const Tensor &output, size_t out_channels,
                           size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                           size_t pad_h, size_t pad_w) {
    auto input_shape = input->shape();
    auto output_shape = output->shape();
    size_t batch_size = input_shape[0];
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];

    size_t expected_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t expected_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], out_channels);
    EXPECT_EQ(output_shape[2], expected_h);
    EXPECT_EQ(output_shape[3], expected_w);
  }

  void verify_forward_result(const Tensor &input, const Tensor &output, const Tensor &weights,
                             const Tensor &bias, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, float tolerance = 1e-4f) {
    const float *input_data = input->data_as<float>();
    const float *output_data = output->data_as<float>();
    const float *weight_data = weights->data_as<float>();
    const float *bias_data = bias != nullptr ? bias->data_as<float>() : nullptr;

    auto input_shape = input->shape();
    auto output_shape = output->shape();
    size_t batch_size = input_shape[0];
    size_t in_channels = input_shape[1];
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];
    size_t out_channels = output_shape[1];
    size_t output_h = output_shape[2];
    size_t output_w = output_shape[3];

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            float expected = bias_data ? bias_data[oc] : 0.0f;

            for (size_t ic = 0; ic < in_channels; ++ic) {
              for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                  int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                  int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);

                  if (ih >= 0 && ih < static_cast<int>(input_h) && iw >= 0 &&
                      iw < static_cast<int>(input_w)) {
                    size_t input_idx = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
                    size_t weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    expected += input_data[input_idx] * weight_data[weight_idx];
                  }
                }
              }
            }

            size_t output_idx = ((n * out_channels + oc) * output_h + oh) * output_w + ow;
            EXPECT_NEAR(output_data[output_idx], expected, tolerance)
                << "Mismatch at batch=" << n << ", out_channel=" << oc << ", oh=" << oh
                << ", ow=" << ow;
          }
        }
      }
    }
  }

  void verify_gradient_shape(const Tensor &gradient, const Tensor &grad_input,
                             const Tensor &original_input) {
    EXPECT_EQ(grad_input->shape(), original_input->shape());
  }

  void verify_backward_result(const Tensor &grad_output, const Tensor &grad_input,
                              const Tensor &weights, size_t kernel_h, size_t kernel_w,
                              size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                              float tolerance = 1e-4f) {
    const float *grad_output_data = grad_output->data_as<float>();
    const float *grad_input_data = grad_input->data_as<float>();
    const float *weight_data = weights->data_as<float>();

    auto grad_input_shape = grad_input->shape();
    auto grad_output_shape = grad_output->shape();
    size_t batch_size = grad_input_shape[0];
    size_t in_channels = grad_input_shape[1];
    size_t input_h = grad_input_shape[2];
    size_t input_w = grad_input_shape[3];
    size_t out_channels = grad_output_shape[1];
    size_t output_h = grad_output_shape[2];
    size_t output_w = grad_output_shape[3];

    std::vector<float> expected_grad_input(grad_input->size(), 0.0f);

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t ic = 0; ic < in_channels; ++ic) {
        for (size_t ih = 0; ih < input_h; ++ih) {
          for (size_t iw = 0; iw < input_w; ++iw) {
            float grad_sum = 0.0f;

            for (size_t oc = 0; oc < out_channels; ++oc) {
              for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                  int oh = (static_cast<int>(ih) + static_cast<int>(pad_h) - static_cast<int>(kh));
                  int ow = (static_cast<int>(iw) + static_cast<int>(pad_w) - static_cast<int>(kw));

                  if (oh >= 0 && ow >= 0 && oh % static_cast<int>(stride_h) == 0 &&
                      ow % static_cast<int>(stride_w) == 0) {
                    oh /= stride_h;
                    ow /= stride_w;

                    if (oh < static_cast<int>(output_h) && ow < static_cast<int>(output_w)) {
                      size_t grad_out_idx =
                          ((n * out_channels + oc) * output_h + oh) * output_w + ow;
                      size_t weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                      grad_sum += grad_output_data[grad_out_idx] * weight_data[weight_idx];
                    }
                  }
                }
              }
            }

            size_t input_idx = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
            expected_grad_input[input_idx] = grad_sum;
          }
        }
      }
    }

    for (size_t i = 0; i < grad_input->size(); ++i) {
      EXPECT_NEAR(grad_input_data[i], expected_grad_input[i], tolerance)
          << "Gradient mismatch at index " << i;
    }
  }

  bool has_cpu_;
};

TEST_F(LegacyConv2DLayerTest, BasicForwardPass) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[2], 3);
  EXPECT_EQ(out_shape[3], 3);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], params.size() > 1 ? params[1] : nullptr, 3, 3, 1,
                        1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithStride) {
  LegacyConv2DLayer layer(1, 2, 3, 3, 2, 2, 0, 0, false, "test_conv_stride");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 7, 7}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 3, 3, 2, 2, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[2], 3);
  EXPECT_EQ(out_shape[3], 3);
  EXPECT_EQ(out_shape[1], 2);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithPadding) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 1, 1, true, "test_conv_padding");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 1, 1);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[2], 5);
  EXPECT_EQ(out_shape[3], 5);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassMultiChannel) {
  LegacyConv2DLayer layer(3, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_multichannel");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 3, 5, 5}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[1], 2);
  EXPECT_EQ(out_shape[2], 3);
  EXPECT_EQ(out_shape[3], 3);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassMultiBatch) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_conv_multibatch");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({4, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[0], 4);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassNonSquareKernel) {
  LegacyConv2DLayer layer(1, 1, 3, 5, 1, 1, 0, 0, true, "test_conv_nonsquare");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 7, 9}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 5, 1, 1, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[2], 5);
  EXPECT_EQ(out_shape[3], 5);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithBias) {
  LegacyConv2DLayer layer(1, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_bias");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[1], 2);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithoutBias) {
  LegacyConv2DLayer layer(1, 2, 3, 3, 1, 1, 0, 0, false, "test_conv_no_bias");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[1], 2);
}

TEST_F(LegacyConv2DLayerTest, BasicBackwardPass) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv_backward");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);

  auto params = layer.parameters();
  verify_backward_result(gradient, grad_input, params[0], 3, 3, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, BackwardPassWithPadding) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 1, 1, true, "test_conv_backward_pad");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyConv2DLayerTest, BackwardPassMultiChannel) {
  LegacyConv2DLayer layer(3, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_backward_multichannel");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 3, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[1], 3);
}

TEST_F(LegacyConv2DLayerTest, BackwardPassMultiBatch) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_conv_backward_multibatch");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({4, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[0], 4);
}

TEST_F(LegacyConv2DLayerTest, BackwardPassVariableGradient) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv_backward_var");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  float *grad_data = gradient->data_as<float>();
  for (size_t i = 0; i < gradient->size(); ++i) {
    grad_data[i] = static_cast<float>(i + 1);
  }

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyConv2DLayerTest, ComputeOutputShape) {
  LegacyConv2DLayer layer(3, 16, 3, 3, 2, 2, 1, 1, true, "test_conv_shape");

  std::vector<size_t> input_shape = {2, 3, 32, 32};
  std::vector<size_t> expected_shape = {2, 16, 16, 16};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyConv2DLayerTest, ComputeOutputShapeWithPadding) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 1, 1, false, "test_conv_shape_pad");

  std::vector<size_t> input_shape = {1, 1, 5, 5};
  std::vector<size_t> expected_shape = {1, 1, 5, 5};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyConv2DLayerTest, GetConfig) {
  LegacyConv2DLayer layer(3, 16, 3, 5, 2, 1, 1, 2, true, "test_conv_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_conv_config");
  EXPECT_EQ(config.get<size_t>("in_channels"), 3);
  EXPECT_EQ(config.get<size_t>("out_channels"), 16);
  EXPECT_EQ(config.get<size_t>("kernel_h"), 3);
  EXPECT_EQ(config.get<size_t>("kernel_w"), 5);
  EXPECT_EQ(config.get<size_t>("stride_h"), 2);
  EXPECT_EQ(config.get<size_t>("stride_w"), 1);
  EXPECT_EQ(config.get<size_t>("pad_h"), 1);
  EXPECT_EQ(config.get<size_t>("pad_w"), 2);
  EXPECT_EQ(config.get<bool>("use_bias"), true);
}

TEST_F(LegacyConv2DLayerTest, Clone) {
  LegacyConv2DLayer original(3, 16, 3, 3, 1, 1, 1, 1, true, "test_conv_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "legacy_conv2d");
  EXPECT_EQ(cloned->type(), original.type());
}

TEST_F(LegacyConv2DLayerTest, EdgeCase1x1Convolution) {
  LegacyConv2DLayer layer(3, 16, 1, 1, 1, 1, 0, 0, true, "test_1x1_conv");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 3, 8, 8}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[2], 8);
  EXPECT_EQ(out_shape[3], 8);
  EXPECT_EQ(out_shape[1], 16);
}

TEST_F(LegacyConv2DLayerTest, EdgeCaseZeroGradient) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_zero_gradient");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(0.0f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyConv2DLayerTest, EdgeCaseLargeValues) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_large_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1e6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output->size(), 9);
}

TEST_F(LegacyConv2DLayerTest, EdgeCaseNegativeValues) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_negative_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, NumericalStabilitySmallValues) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_small_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1e-6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output->size(), 9);
}

TEST_F(LegacyConv2DLayerTest, BackwardNumericalStability) {
  LegacyConv2DLayer layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_backward_stability");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 1, 5, 5}, getCPU());
  input->fill(1e-6f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1e-6f);

  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(LegacyConv2DLayerTest, ParameterCollectionWithBias) {
  LegacyConv2DLayer layer(3, 16, 3, 3, 1, 1, 0, 0, true, "test_params_bias");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> params = layer.parameters();

  EXPECT_EQ(params.size(), 2);
}

TEST_F(LegacyConv2DLayerTest, ParameterCollectionWithoutBias) {
  LegacyConv2DLayer layer(3, 16, 3, 3, 1, 1, 0, 0, false, "test_params_no_bias");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> params = layer.parameters();
  EXPECT_EQ(params.size(), 1);
}

TEST_F(LegacyConv2DLayerTest, GradientCollectionWithBias) {
  LegacyConv2DLayer layer(3, 16, 3, 3, 1, 1, 0, 0, true, "test_grads_bias");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 2);
}

TEST_F(LegacyConv2DLayerTest, GradientCollectionWithoutBias) {
  LegacyConv2DLayer layer(3, 16, 3, 3, 1, 1, 0, 0, false, "test_grads_no_bias");
  layer.set_device(getCPU());
  layer.init();

  std::vector<Tensor> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 1);
}

TEST_F(LegacyConv2DLayerTest, ResNet1x1ChannelIncrease) {
  LegacyConv2DLayer layer(64, 256, 1, 1, 1, 1, 0, 0, false, "resnet_1x1_increase");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 64, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 256);
  EXPECT_EQ(output_shape_actual[2], 8);
  EXPECT_EQ(output_shape_actual[3], 8);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 1, 1, 1, 1, 0, 0);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 1, 1, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNet1x1ChannelDecrease) {
  LegacyConv2DLayer layer(256, 64, 1, 1, 1, 1, 0, 0, false, "resnet_1x1_decrease");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 256, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 50) * 0.02f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto out_shape = output->shape();
  EXPECT_EQ(out_shape[0], 2);
  EXPECT_EQ(out_shape[1], 64);
  EXPECT_EQ(out_shape[2], 8);
  EXPECT_EQ(out_shape[3], 8);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 1, 1, 1, 1, 0, 0);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 1, 1, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNetStridedDownsample) {
  LegacyConv2DLayer layer(64, 128, 3, 3, 2, 2, 0, 0, false, "resnet_strided_downsample");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 64, 9, 9}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 128);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 3, 3, 2, 2, 0, 0);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 3, 3, 2, 2, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNetStridedWithPadding) {
  LegacyConv2DLayer layer(64, 128, 3, 3, 2, 2, 1, 1, false, "resnet_strided_padded");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 64, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 128);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 3, 3, 2, 2, 1, 1);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 3, 3, 2, 2, 1, 1);
}

TEST_F(LegacyConv2DLayerTest, ResNet1x1StridedDownsample) {
  LegacyConv2DLayer layer(64, 256, 1, 1, 2, 2, 0, 0, false, "resnet_1x1_strided");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 64, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 256);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 1, 1, 2, 2, 0, 0);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 1, 1, 2, 2, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNetBottleneck3x3) {
  LegacyConv2DLayer layer(64, 64, 3, 3, 1, 1, 1, 1, false, "resnet_bottleneck_3x3");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 64, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 8);
  EXPECT_EQ(output_shape_actual[3], 8);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 3, 3, 1, 1, 1, 1);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 3, 3, 1, 1, 1, 1);
}

TEST_F(LegacyConv2DLayerTest, ResNetFirstConv7x7) {
  LegacyConv2DLayer layer(3, 64, 7, 7, 2, 2, 3, 3, true, "resnet_first_conv");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 3, 15, 15}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 256) / 255.0f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 8);
  EXPECT_EQ(output_shape_actual[3], 8);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], params.size() > 1 ? params[1] : nullptr, 7, 7, 2,
                        2, 3, 3);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(0.01f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 7, 7, 2, 2, 3, 3);
}

TEST_F(LegacyConv2DLayerTest, ResNetAsymmetricStride) {
  LegacyConv2DLayer layer(32, 64, 3, 3, 2, 1, 1, 1, false, "resnet_asymmetric_stride");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({1, 32, 8, 8}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 1);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 8);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 3, 3, 2, 1, 1, 1);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 3, 3, 2, 1, 1, 1);
}

TEST_F(LegacyConv2DLayerTest, ResNetSmallFeatureMap) {
  LegacyConv2DLayer layer(64, 64, 3, 3, 2, 2, 1, 1, false, "resnet_small_feature");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = make_tensor<float>({2, 64, 7, 7}, getCPU());
  float *input_data = input->data_as<float>();
  for (size_t i = 0; i < input->size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = make_tensor<float>(output_shape, getCPU());
  layer.forward(input, output);

  auto output_shape_actual = output->shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  auto params = layer.parameters();
  verify_forward_result(input, output, params[0], nullptr, 3, 3, 2, 2, 1, 1);

  Tensor gradient = make_tensor<float>(output->shape(), getCPU());
  gradient->fill(1.0f);
  Tensor grad_input = make_tensor<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  EXPECT_EQ(grad_input->shape(), input->shape());
  verify_backward_result(gradient, grad_input, params[0], 3, 3, 2, 2, 1, 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
