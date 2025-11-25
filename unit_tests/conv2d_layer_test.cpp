/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace tnn;

/**
 * Test fixture for Conv2DLayer validation tests.
 * These tests verify the mathematical correctness of 2D convolution operations
 * including forward and backward passes.
 */
class Conv2DLayerTest : public ::testing::Test {
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

  // Verify forward pass output shape
  void verify_output_shape(const Tensor<float> &input, const Tensor<float> &output,
                           size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                           size_t stride_w, size_t pad_h, size_t pad_w) {
    size_t batch_size = input.batch_size();
    size_t input_h = input.height();
    size_t input_w = input.width();

    size_t expected_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t expected_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    EXPECT_EQ(output.batch_size(), batch_size);
    EXPECT_EQ(output.channels(), out_channels);
    EXPECT_EQ(output.height(), expected_h);
    EXPECT_EQ(output.width(), expected_w);
  }

  // Verify forward pass numerical correctness
  void verify_forward_result(const Tensor<float> &input, const Tensor<float> &output,
                             const Tensor<float> &weights, const Tensor<float> *bias,
                             size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                             size_t pad_h, size_t pad_w, float tolerance = 1e-4f) {
    const float *input_data = input.data();
    const float *output_data = output.data();
    const float *weight_data = weights.data();
    const float *bias_data = bias ? bias->data() : nullptr;

    size_t batch_size = input.batch_size();
    size_t in_channels = input.channels();
    size_t input_h = input.height();
    size_t input_w = input.width();
    size_t out_channels = output.channels();
    size_t output_h = output.height();
    size_t output_w = output.width();

    // For each output element, compute expected value via manual convolution
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            float expected = bias_data ? bias_data[oc] : 0.0f;

            // Convolve over all input channels and kernel positions
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

  // Verify backward pass gradient shape
  void verify_gradient_shape(const Tensor<float> &gradient, const Tensor<float> &grad_input,
                             const Tensor<float> &original_input) {
    EXPECT_EQ(grad_input.shape(), original_input.shape());
  }

  // Verify backward pass numerical correctness for input gradients
  void verify_backward_result(const Tensor<float> &grad_output, const Tensor<float> &grad_input,
                              const Tensor<float> &weights, size_t kernel_h, size_t kernel_w,
                              size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                              float tolerance = 1e-4f) {
    const float *grad_output_data = grad_output.data();
    const float *grad_input_data = grad_input.data();
    const float *weight_data = weights.data();

    size_t batch_size = grad_input.batch_size();
    size_t in_channels = grad_input.channels();
    size_t input_h = grad_input.height();
    size_t input_w = grad_input.width();
    size_t out_channels = grad_output.channels();
    size_t output_h = grad_output.height();
    size_t output_w = grad_output.width();

    std::vector<float> expected_grad_input(grad_input.size(), 0.0f);

    // Compute expected gradient via transposed convolution
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t ic = 0; ic < in_channels; ++ic) {
        for (size_t ih = 0; ih < input_h; ++ih) {
          for (size_t iw = 0; iw < input_w; ++iw) {
            float grad_sum = 0.0f;

            // For each output position that could have used this input position
            for (size_t oc = 0; oc < out_channels; ++oc) {
              for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                  // Check if this kernel position at this output channel used input[ih][iw]
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
  const Device *cpu_device_;
};

// Forward Pass Tests

TEST_F(Conv2DLayerTest, BasicForwardPass) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.height(), 3);
  EXPECT_EQ(output.width(), 3);

  // Verify numerical correctness
  auto params = layer.parameters();
  verify_forward_result(input, output, *params[0], params.size() > 1 ? params[1] : nullptr, 3, 3, 1,
                        1, 0, 0);
}

TEST_F(Conv2DLayerTest, ForwardPassWithStride) {
  Conv2DLayer<float> layer(1, 2, 3, 3, 2, 2, 0, 0, false, "test_conv_stride");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 7, 7}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 2, 3, 3, 2, 2, 0, 0);
  EXPECT_EQ(output.height(), 3);
  EXPECT_EQ(output.width(), 3);
  EXPECT_EQ(output.channels(), 2);
}

TEST_F(Conv2DLayerTest, ForwardPassWithPadding) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 1, 1, true, "test_conv_padding");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 1, 1);
  EXPECT_EQ(output.height(), 5);
  EXPECT_EQ(output.width(), 5);
}

TEST_F(Conv2DLayerTest, ForwardPassMultiChannel) {
  Conv2DLayer<float> layer(3, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_multichannel");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 3, 5, 5}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = static_cast<float>(i % 10);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.channels(), 2);
  EXPECT_EQ(output.height(), 3);
  EXPECT_EQ(output.width(), 3);
}

TEST_F(Conv2DLayerTest, ForwardPassMultiBatch) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_conv_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({4, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.batch_size(), 4);
}

TEST_F(Conv2DLayerTest, ForwardPassNonSquareKernel) {
  Conv2DLayer<float> layer(1, 1, 3, 5, 1, 1, 0, 0, true, "test_conv_nonsquare");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 7, 9}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 5, 1, 1, 0, 0);
  EXPECT_EQ(output.height(), 5);
  EXPECT_EQ(output.width(), 5);
}

TEST_F(Conv2DLayerTest, ForwardPassWithBias) {
  Conv2DLayer<float> layer(1, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.channels(), 2);
}

TEST_F(Conv2DLayerTest, ForwardPassWithoutBias) {
  Conv2DLayer<float> layer(1, 2, 3, 3, 1, 1, 0, 0, false, "test_conv_no_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 2, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.channels(), 2);
}

// Backward Pass Tests

TEST_F(Conv2DLayerTest, BasicBackwardPass) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv_backward");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);

  // Verify numerical correctness
  auto params = layer.parameters();
  verify_backward_result(gradient, grad_input, *params[0], 3, 3, 1, 1, 0, 0);
}

TEST_F(Conv2DLayerTest, BackwardPassWithPadding) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 1, 1, true, "test_conv_backward_pad");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(Conv2DLayerTest, BackwardPassMultiChannel) {
  Conv2DLayer<float> layer(3, 2, 3, 3, 1, 1, 0, 0, true, "test_conv_backward_multichannel");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 3, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.channels(), 3);
}

TEST_F(Conv2DLayerTest, BackwardPassMultiBatch) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_conv_backward_multibatch");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({4, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input.batch_size(), 4);
}

TEST_F(Conv2DLayerTest, BackwardPassVariableGradient) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_conv_backward_var");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
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

// Configuration Tests

TEST_F(Conv2DLayerTest, ComputeOutputShape) {
  Conv2DLayer<float> layer(3, 16, 3, 3, 2, 2, 1, 1, true, "test_conv_shape");

  std::vector<size_t> input_shape = {2, 3, 32, 32};
  std::vector<size_t> expected_shape = {2, 16, 16, 16};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(Conv2DLayerTest, ComputeOutputShapeWithPadding) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 1, 1, false, "test_conv_shape_pad");

  std::vector<size_t> input_shape = {1, 1, 5, 5};
  std::vector<size_t> expected_shape = {1, 1, 5, 5};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(Conv2DLayerTest, GetConfig) {
  Conv2DLayer<float> layer(3, 16, 3, 5, 2, 1, 1, 2, true, "test_conv_config");

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

TEST_F(Conv2DLayerTest, Clone) {
  Conv2DLayer<float> original(3, 16, 3, 3, 1, 1, 1, 1, true, "test_conv_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "conv2d");
  EXPECT_EQ(cloned->type(), original.type());
}

// Edge Cases

TEST_F(Conv2DLayerTest, EdgeCase1x1Convolution) {
  Conv2DLayer<float> layer(3, 16, 1, 1, 1, 1, 0, 0, true, "test_1x1_conv");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 3, 8, 8}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  EXPECT_EQ(output.height(), 8);
  EXPECT_EQ(output.width(), 8);
  EXPECT_EQ(output.channels(), 16);
}

TEST_F(Conv2DLayerTest, EdgeCaseZeroGradient) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_zero_gradient");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1.0f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(0.0f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST_F(Conv2DLayerTest, EdgeCaseLargeValues) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_large_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1e6f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.size(), 9);
}

TEST_F(Conv2DLayerTest, EdgeCaseNegativeValues) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_negative_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  float *input_data = input.data();
  for (size_t i = 0; i < input.size(); ++i) {
    input_data[i] = -static_cast<float>(i + 1);
  }

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
}

// Numerical Stability Tests

TEST_F(Conv2DLayerTest, NumericalStabilitySmallValues) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, true, "test_small_values");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1e-6f);

  const Tensor<float> &output = layer.forward(input);

  verify_output_shape(input, output, 1, 3, 3, 1, 1, 0, 0);
  EXPECT_EQ(output.size(), 9);
}

TEST_F(Conv2DLayerTest, BackwardNumericalStability) {
  Conv2DLayer<float> layer(1, 1, 3, 3, 1, 1, 0, 0, false, "test_backward_stability");
  layer.set_device(cpu_device_);
  layer.initialize();

  Tensor<float> input({1, 1, 5, 5}, cpu_device_);
  input.fill(1e-6f);

  const Tensor<float> &output = layer.forward(input);

  Tensor<float> gradient(output.shape(), cpu_device_);
  gradient.fill(1e-6f);

  const Tensor<float> &grad_input = layer.backward(gradient);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(Conv2DLayerTest, ParameterCollectionWithBias) {
  Conv2DLayer<float> layer(3, 16, 3, 3, 1, 1, 0, 0, true, "test_params_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> params = layer.parameters();

  EXPECT_EQ(params.size(), 2); // weights and bias
}

TEST_F(Conv2DLayerTest, ParameterCollectionWithoutBias) {
  Conv2DLayer<float> layer(3, 16, 3, 3, 1, 1, 0, 0, false, "test_params_no_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> params = layer.parameters();
  EXPECT_EQ(params.size(), 1); // weights only
}

TEST_F(Conv2DLayerTest, GradientCollectionWithBias) {
  Conv2DLayer<float> layer(3, 16, 3, 3, 1, 1, 0, 0, true, "test_grads_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 2); // weight gradients and bias gradients
}

TEST_F(Conv2DLayerTest, GradientCollectionWithoutBias) {
  Conv2DLayer<float> layer(3, 16, 3, 3, 1, 1, 0, 0, false, "test_grads_no_bias");
  layer.set_device(cpu_device_);
  layer.initialize();

  std::vector<Tensor<float> *> grads = layer.gradients();

  EXPECT_EQ(grads.size(), 1); // weight gradients only
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
