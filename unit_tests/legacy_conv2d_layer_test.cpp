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
#include "device/pool_allocator.hpp"
#include "tensor/tensor.hpp"
#include "test_graph_utils.hpp"

using namespace synet;

/**
 * Test fixture for LegacyConv2DLayerImpl validation tests.
 * These tests verify the mathematical correctness of 2D convolution operations
 * including forward and backward passes.
 */
class LegacyConv2DLayerTest : public ::testing::Test {
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

  void verify_output_shape(const Tensor &input, Tensor &output, size_t out_channels,
                           size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                           size_t pad_h, size_t pad_w) {
    auto input_shape = input.shape();
    auto output_shape = output.shape();
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

  void verify_forward_result(const Tensor &input, Tensor &output, const Tensor &weights,
                             const Tensor &bias, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, float tolerance = 1e-4f) {
    const float *input_data = input.data_as<float>();
    const float *output_data = output.data_as<float>();
    const float *weight_data = weights.data_as<float>();
    const float *bias_data = bias ? bias.data_as<float>() : nullptr;

    auto input_shape = input.shape();
    auto output_shape = output.shape();
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

  void verify_gradient_shape(const Tensor &grad_output, Tensor &grad_input,
                             const Tensor &original_input) {
    EXPECT_EQ(grad_input.shape(), original_input.shape());
  }

  void verify_backward_result(const Tensor &grad_output, Tensor &grad_input, const Tensor &weights,
                              size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                              size_t pad_h, size_t pad_w, float tolerance = 1e-4f) {
    const float *grad_output_data = grad_output.data_as<float>();
    const float *grad_input_data = grad_input.data_as<float>();
    const float *weight_data = weights.data_as<float>();

    auto grad_input_shape = grad_input.shape();
    auto grad_output_shape = grad_output.shape();
    size_t batch_size = grad_input_shape[0];
    size_t in_channels = grad_input_shape[1];
    size_t input_h = grad_input_shape[2];
    size_t input_w = grad_input_shape[3];
    size_t out_channels = grad_output_shape[1];
    size_t output_h = grad_output_shape[2];
    size_t output_w = grad_output_shape[3];

    Vec<float> expected_grad_input(grad_input.size(), 0.0f);

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

    for (size_t i = 0; i < grad_input.size(); ++i) {
      EXPECT_NEAR(grad_input_data[i], expected_grad_input[i], tolerance)
          << "Gradient mismatch at index " << i;
    }
  }

  bool has_cpu_;
};

TEST_F(LegacyConv2DLayerTest, BasicForwardPass) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[2], 3);
  EXPECT_EQ(out_shape[3], 3);

  auto params = layer.parameters();
  verify_forward_result(input, output, *params[0], params.size() > 1 ? *params[1] : Tensor(), 3, 3,
                        1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithStride) {
  auto layer = LegacyConv2DLayer(1, 2, 3, 3, 2, 2, 0, 0, false, "test_conv_stride");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 7, 7}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 2, 3, 3, 2, 2, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[2], 3);
  EXPECT_EQ(out_shape[3], 3);
  EXPECT_EQ(out_shape[1], 2);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithPadding) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 1, 1, true, "test_conv_padding");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 1, 1);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[2], 5);
  EXPECT_EQ(out_shape[3], 5);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassMultiChannel) {
  auto layer = LegacyConv2DLayer(3, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_multichannel");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 3, 5, 5}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[1], 2);
  EXPECT_EQ(out_shape[2], 3);
  EXPECT_EQ(out_shape[3], 3);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassMultiBatch) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_conv_multibatch");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({4, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[0], 4);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassNonSquareKernel) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 5, 1, 1, 0, 0, true, "test_conv_nonsquare");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 7, 9}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 5, 1, 1, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[2], 5);
  EXPECT_EQ(out_shape[3], 5);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithBias) {
  auto layer = LegacyConv2DLayer(1, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_bias");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[1], 2);
}

TEST_F(LegacyConv2DLayerTest, ForwardPassWithoutBias) {
  auto layer = LegacyConv2DLayer(1, 2, 3, 3, 1, 1, 0, 0, false, "test_conv_no_bias");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[1], 2);
}

TEST_F(LegacyConv2DLayerTest, BasicBackwardPass) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv_backward");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);

  auto params = layer.parameters();
  verify_backward_result(grad_output, grad_input, *params[0], 3, 3, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, BackwardPassWithPadding) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 1, 1, true, "test_conv_backward_pad");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(LegacyConv2DLayerTest, BackwardPassMultiChannel) {
  auto layer = LegacyConv2DLayer(3, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_backward_multichannel");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 3, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  auto grad_input_shape = grad_input.shape();
  EXPECT_EQ(grad_input_shape[1], 3);
}

TEST_F(LegacyConv2DLayerTest, BackwardPassMultiBatch) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_conv_backward_multibatch");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({4, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
  auto grad_input_shape = grad_input.shape();
  EXPECT_EQ(grad_input_shape[0], 4);
}

TEST_F(LegacyConv2DLayerTest, BackwardPassVariableGradient) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv_backward_var");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
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

TEST_F(LegacyConv2DLayerTest, ComputeOutputShape) {
  auto layer = LegacyConv2DLayer(3, 16, 3, 3, 2, 2, 1, 1, true, "test_conv_shape");

  Vec<size_t> input_shape = {2, 3, 32, 32};
  Vec<size_t> expected_shape = {2, 16, 16, 16};

  Vec<size_t> output_shape = layer.output_shapes({input_shape})[0];

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyConv2DLayerTest, ComputeOutputShapeWithPadding) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 1, 1, false, "test_conv_shape_pad");

  Vec<size_t> input_shape = {1, 1, 5, 5};
  Vec<size_t> expected_shape = {1, 1, 5, 5};

  Vec<size_t> output_shape = layer.output_shapes({input_shape})[0];

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyConv2DLayerTest, GetConfig) {
  auto layer = LegacyConv2DLayer(3, 16, 3, 5, 2, 1, 1, 2, true, "test_conv_config");

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

TEST_F(LegacyConv2DLayerTest, EdgeCase1x1Convolution) {
  auto layer = LegacyConv2DLayer(3, 16, 1, 1, 1, 1, 0, 0, true, "test_1x1_conv");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 3, 8, 8}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[2], 8);
  EXPECT_EQ(out_shape[3], 8);
  EXPECT_EQ(out_shape[1], 16);
}

TEST_F(LegacyConv2DLayerTest, EdgeCaseZeroGradient) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_zero_gradient");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1.0f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(0.0f);

  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(LegacyConv2DLayerTest, EdgeCaseLargeValues) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_large_values");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1e6f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.size(), 9);
}

TEST_F(LegacyConv2DLayerTest, EdgeCaseNegativeValues) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_negative_values");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, NumericalStabilitySmallValues) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_small_values");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1e-6f);

  Tensor output = layer.forward({input})[0];

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.size(), 9);
}

TEST_F(LegacyConv2DLayerTest, BackwardNumericalStability) {
  auto layer = LegacyConv2DLayer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_backward_stability");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 1, 5, 5}, DType_t::FP32, getHost());
  input.fill(1e-6f);

  Tensor output = layer.forward({input})[0];

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1e-6f);

  Tensor grad_input = layer.backward({grad_output})[0];

  verify_gradient_shape(grad_output, grad_input, input);
}

TEST_F(LegacyConv2DLayerTest, ParameterCollectionWithBias) {
  auto layer = LegacyConv2DLayer(3, 16, 3, 3, 1, 1, 0, 0, true, "test_params_bias");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Vec<Tensor *> params = layer.parameters();

  EXPECT_EQ(params.size(), 2);
}

TEST_F(LegacyConv2DLayerTest, ParameterCollectionWithoutBias) {
  auto layer = LegacyConv2DLayer(3, 16, 3, 3, 1, 1, 0, 0, false, "test_params_no_bias");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Vec<Tensor *> params = layer.parameters();
  EXPECT_EQ(params.size(), 1);
}

TEST_F(LegacyConv2DLayerTest, GradientCollectionWithBias) {
  auto layer = LegacyConv2DLayer(3, 16, 3, 3, 1, 1, 0, 0, true, "test_grads_bias");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Vec<Tensor *> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 2);
}

TEST_F(LegacyConv2DLayerTest, GradientCollectionWithoutBias) {
  auto layer = LegacyConv2DLayer(3, 16, 3, 3, 1, 1, 0, 0, false, "test_grads_no_bias");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Vec<Tensor *> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 1);
}

TEST_F(LegacyConv2DLayerTest, ResNet1x1ChannelIncrease) {
  auto layer = LegacyConv2DLayer(64, 256, 1, 1, 1, 1, 0, 0, false, "resnet_1x1_increase");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 64, 8, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 256);
  EXPECT_EQ(output_shape_actual[2], 8);
  EXPECT_EQ(output_shape_actual[3], 8);

  auto params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 1, 1, 1, 1, 0, 0);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 1, 1, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNet1x1ChannelDecrease) {
  auto layer = LegacyConv2DLayer(256, 64, 1, 1, 1, 1, 0, 0, false, "resnet_1x1_decrease");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 256, 8, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 50) * 0.02f);
  }

  Tensor output = layer.forward({input})[0];

  auto out_shape = output.shape();
  EXPECT_EQ(out_shape[0], 2);
  EXPECT_EQ(out_shape[1], 64);
  EXPECT_EQ(out_shape[2], 8);
  EXPECT_EQ(out_shape[3], 8);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 1, 1, 1, 1, 0, 0);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 1, 1, 1, 1, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNetStridedDownsample) {
  auto layer = LegacyConv2DLayer(64, 128, 3, 3, 2, 2, 0, 0, false, "resnet_strided_downsample");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 64, 9, 9}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 128);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 3, 3, 2, 2, 0, 0);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 3, 3, 2, 2, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNetStridedWithPadding) {
  auto layer = LegacyConv2DLayer(64, 128, 3, 3, 2, 2, 1, 1, false, "resnet_strided_padded");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 64, 8, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 128);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 3, 3, 2, 2, 1, 1);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 3, 3, 2, 2, 1, 1);
}

TEST_F(LegacyConv2DLayerTest, ResNet1x1StridedDownsample) {
  auto layer = LegacyConv2DLayer(64, 256, 1, 1, 2, 2, 0, 0, false, "resnet_1x1_strided");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 64, 8, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 256);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 1, 1, 2, 2, 0, 0);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 1, 1, 2, 2, 0, 0);
}

TEST_F(LegacyConv2DLayerTest, ResNetBottleneck3x3) {
  auto layer = LegacyConv2DLayer(64, 64, 3, 3, 1, 1, 1, 1, false, "resnet_bottleneck_3x3");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 64, 8, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 8);
  EXPECT_EQ(output_shape_actual[3], 8);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 3, 3, 1, 1, 1, 1);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 3, 3, 1, 1, 1, 1);
}

TEST_F(LegacyConv2DLayerTest, ResNetFirstConv7x7) {
  auto layer = LegacyConv2DLayer(3, 64, 7, 7, 2, 2, 3, 3, true, "resnet_first_conv");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 3, 15, 15}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 256) / 255.0f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 8);
  EXPECT_EQ(output_shape_actual[3], 8);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], params.size() > 1 ? *params[1] : Tensor(), 7, 7,
                        2, 2, 3, 3);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(0.01f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 7, 7, 2, 2, 3, 3);
}

TEST_F(LegacyConv2DLayerTest, ResNetAsymmetricStride) {
  auto layer = LegacyConv2DLayer(32, 64, 3, 3, 2, 1, 1, 1, false, "resnet_asymmetric_stride");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({1, 32, 8, 8}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 1);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 8);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 3, 3, 2, 1, 1, 1);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 3, 3, 2, 1, 1, 1);
}

TEST_F(LegacyConv2DLayerTest, ResNetSmallFeatureMap) {
  auto layer = LegacyConv2DLayer(64, 64, 3, 3, 2, 2, 1, 1, false, "resnet_small_feature");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  auto graph = test::compile_single_layer(layer, allocator);

  Tensor input = Tensor({2, 64, 7, 7}, DType_t::FP32, getHost());
  float *input_data = input.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) * 0.01f);
  }

  Tensor output = layer.forward({input})[0];

  auto output_shape_actual = output.shape();
  EXPECT_EQ(output_shape_actual[0], 2);
  EXPECT_EQ(output_shape_actual[1], 64);
  EXPECT_EQ(output_shape_actual[2], 4);
  EXPECT_EQ(output_shape_actual[3], 4);

  Vec<Tensor *> params = layer.parameters();
  verify_forward_result(input, output, *params[0], Tensor(), 3, 3, 2, 2, 1, 1);

  Tensor grad_output = Tensor(output.shape(), DType_t::FP32, getHost());
  grad_output.fill(1.0f);
  Tensor grad_input = layer.backward({grad_output})[0];

  EXPECT_EQ(grad_input.shape(), input.shape());
  verify_backward_result(grad_output, grad_input, *params[0], 3, 3, 2, 2, 1, 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
