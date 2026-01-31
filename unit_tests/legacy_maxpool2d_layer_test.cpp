/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/legacy_maxpool2d_layer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "device/device_manager.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Test fixture for LegacyMaxPool2DLayer validation tests.
 * These tests verify the mathematical correctness of max pooling operations
 * including forward and backward passes.
 */
class LegacyMaxPool2DLayerTest : public ::testing::Test {
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

  // Verify forward pass output shape
  void verify_output_shape(const Tensor &input, const Tensor &output, size_t pool_h, size_t pool_w,
                           size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
    auto input_shape = input->shape();
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];

    size_t expected_h = (input_h + 2 * pad_h - pool_h) / stride_h + 1;
    size_t expected_w = (input_w + 2 * pad_w - pool_w) / stride_w + 1;

    auto output_shape = output->shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], channels);
    EXPECT_EQ(output_shape[2], expected_h);
    EXPECT_EQ(output_shape[3], expected_w);
  }

  // Verify backward pass gradient shape
  void verify_gradient_shape(const Tensor &gradient, const Tensor &grad_input,
                             const Tensor &original_input) {
    auto grad_input_shape = grad_input->shape();
    auto original_input_shape = original_input->shape();
    EXPECT_EQ(grad_input_shape[0], original_input_shape[0]);
    EXPECT_EQ(grad_input_shape[1], original_input_shape[1]);
    EXPECT_EQ(grad_input_shape[2], original_input_shape[2]);
    EXPECT_EQ(grad_input_shape[3], original_input_shape[3]);
  }

  // Verify backward pass numerical correctness
  void verify_backward_result(const Tensor &input, const Tensor &grad_output,
                              const Tensor &grad_input, size_t pool_h, size_t pool_w,
                              size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                              float tolerance = 1e-5f) {
    const float *input_data = input->data_as<float>();
    const float *grad_output_data = grad_output->data_as<float>();
    const float *grad_input_data = grad_input->data_as<float>();

    auto input_shape = input->shape();
    auto grad_output_shape = grad_output->shape();
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];
    size_t output_h = grad_output_shape[2];
    size_t output_w = grad_output_shape[3];

    std::vector<float> expected_grad_input(grad_input->size(), 0.0f);

    // For each output position, route gradient to the max input position
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            // Find the max position in the pooling window
            float max_val = -std::numeric_limits<float>::infinity();
            int max_ih = -1, max_iw = -1;

            for (size_t ph = 0; ph < pool_h; ++ph) {
              for (size_t pw = 0; pw < pool_w; ++pw) {
                int ih = static_cast<int>(oh * stride_h + ph) - static_cast<int>(pad_h);
                int iw = static_cast<int>(ow * stride_w + pw) - static_cast<int>(pad_w);

                if (ih >= 0 && ih < static_cast<int>(input_h) && iw >= 0 &&
                    iw < static_cast<int>(input_w)) {
                  size_t input_idx = ((n * channels + c) * input_h + ih) * input_w + iw;
                  if (input_data[input_idx] > max_val) {
                    max_val = input_data[input_idx];
                    max_ih = ih;
                    max_iw = iw;
                  }
                }
              }
            }

            // Route gradient to max position
            if (max_ih >= 0 && max_iw >= 0) {
              size_t max_input_idx = ((n * channels + c) * input_h + max_ih) * input_w + max_iw;
              size_t grad_output_idx = ((n * channels + c) * output_h + oh) * output_w + ow;
              expected_grad_input[max_input_idx] += grad_output_data[grad_output_idx];
            }
          }
        }
      }
    }

    for (size_t i = 0; i < grad_input->size(); ++i) {
      EXPECT_NEAR(grad_input_data[i], expected_grad_input[i], tolerance)
          << "Gradient mismatch at index " << i;
    }
  }

  // Verify that max values are correctly selected
  void verify_max_selection(const Tensor &input, const Tensor &output, size_t pool_h, size_t pool_w,
                            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                            float tolerance = 1e-5f) {
    const float *input_data = input->data_as<float>();
    const float *output_data = output->data_as<float>();

    auto input_shape = input->shape();
    auto output_shape = output->shape();
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t input_h = input_shape[2];
    size_t input_w = input_shape[3];
    size_t output_h = output_shape[2];
    size_t output_w = output_shape[3];

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            float max_val = -std::numeric_limits<float>::infinity();

            for (size_t ph = 0; ph < pool_h; ++ph) {
              for (size_t pw = 0; pw < pool_w; ++pw) {
                int ih = static_cast<int>(oh * stride_h + ph) - static_cast<int>(pad_h);
                int iw = static_cast<int>(ow * stride_w + pw) - static_cast<int>(pad_w);

                if (ih >= 0 && ih < static_cast<int>(input_h) && iw >= 0 &&
                    iw < static_cast<int>(input_w)) {
                  size_t input_idx = ((n * channels + c) * input_h + ih) * input_w + iw;
                  max_val = std::max(max_val, input_data[input_idx]);
                }
              }
            }

            size_t output_idx = ((n * channels + c) * output_h + oh) * output_w + ow;
            EXPECT_NEAR(output_data[output_idx], max_val, tolerance)
                << "Max value mismatch at batch=" << n << ", channel=" << c << ", oh=" << oh
                << ", ow=" << ow;
          }
        }
      }
    }
  }

  bool has_cpu_;
};

// Forward Pass Tests

TEST_F(LegacyMaxPool2DLayerTest, BasicForwardPass) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  verify_max_selection(input, output, 2, 2, 2, 2, 0, 0);

  const float *output_data = output->data_as<float>();
  EXPECT_NEAR(output_data[0], 6.0f, 1e-5f);   // max of [1,2,5,6]
  EXPECT_NEAR(output_data[1], 8.0f, 1e-5f);   // max of [3,4,7,8]
  EXPECT_NEAR(output_data[2], 14.0f, 1e-5f);  // max of [9,10,13,14]
  EXPECT_NEAR(output_data[3], 16.0f, 1e-5f);  // max of [11,12,15,16]
}

TEST_F(LegacyMaxPool2DLayerTest, ForwardPassWithStride) {
  LegacyMaxPool2DLayer layer(3, 3, 1, 1, 0, 0, "test_maxpool_stride");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 5, 5}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 25; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 3, 3, 1, 1, 0, 0);
  verify_max_selection(input, output, 3, 3, 1, 1, 0, 0);
}

TEST_F(LegacyMaxPool2DLayerTest, ForwardPassWithPadding) {
  LegacyMaxPool2DLayer layer(3, 3, 1, 1, 1, 1, "test_maxpool_padding");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 3, 3}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 9; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 3, 3, 1, 1, 1, 1);
  EXPECT_EQ(output_shape[2], 3);
  EXPECT_EQ(output_shape[3], 3);
}

TEST_F(LegacyMaxPool2DLayerTest, ForwardPassMultiChannel) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_multichannel");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 2, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  EXPECT_EQ(output_shape[1], 2);
  EXPECT_EQ(output_shape[2], 2);
  EXPECT_EQ(output_shape[3], 2);
}

TEST_F(LegacyMaxPool2DLayerTest, ForwardPassMultiBatch) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_multibatch");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({2, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  EXPECT_EQ(output_shape[0], 2);
}

TEST_F(LegacyMaxPool2DLayerTest, ForwardPassNonSquarePooling) {
  LegacyMaxPool2DLayer layer(3, 2, 2, 2, 0, 0, "test_maxpool_nonsquare");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 6, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 24; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 3, 2, 2, 2, 0, 0);
}

TEST_F(LegacyMaxPool2DLayerTest, ForwardPassUniformValues) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_uniform");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  input->fill(5.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);

  const float *output_data = output->data_as<float>();
  for (size_t i = 0; i < output->size(); ++i) {
    EXPECT_NEAR(output_data[i], 5.0f, 1e-5f);
  }
}

// Backward Pass Tests

TEST_F(LegacyMaxPool2DLayerTest, BasicBackwardPass) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_backward");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>({1, 1, 2, 2}, getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input->shape(), input->shape());

  // Verify numerical correctness
  verify_backward_result(input, gradient, grad_input, 2, 2, 2, 2, 0, 0);
}

TEST_F(LegacyMaxPool2DLayerTest, BackwardPassWithPadding) {
  LegacyMaxPool2DLayer layer(3, 3, 1, 1, 1, 1, "test_maxpool_backward_pad");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 3, 3}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 9; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyMaxPool2DLayerTest, BackwardPassMultiChannel) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_backward_multichannel");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 2, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[1], 2);
}

TEST_F(LegacyMaxPool2DLayerTest, BackwardPassMultiBatch) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_backward_multibatch");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({2, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 32; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), getCPU());
  gradient->fill(1.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  auto grad_input_shape = grad_input->shape();
  EXPECT_EQ(grad_input_shape[0], 2);
}

TEST_F(LegacyMaxPool2DLayerTest, BackwardPassVariableGradient) {
  LegacyMaxPool2DLayer layer(2, 2, 1, 1, 0, 0, "test_maxpool_backward_var");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 3, 3}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 9; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), getCPU());
  float *grad_data = gradient->data_as<float>();
  for (size_t i = 0; i < gradient->size(); ++i) {
    grad_data[i] = static_cast<float>(i + 1);
  }

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
  EXPECT_EQ(grad_input->shape(), input->shape());
}

TEST_F(LegacyMaxPool2DLayerTest, BackwardPassGradientRouting) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_grad_routing");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>({1, 1, 2, 2}, getCPU());
  float *grad_data = gradient->data_as<float>();
  grad_data[0] = 1.0f;
  grad_data[1] = 2.0f;
  grad_data[2] = 3.0f;
  grad_data[3] = 4.0f;

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);

  // Verify that gradients are routed to max positions only
  const float *grad_input_data = grad_input->data_as<float>();

  // Position 5 (value 6) should get gradient 1.0
  EXPECT_NEAR(grad_input_data[5], 1.0f, 1e-5f);

  // Position 7 (value 8) should get gradient 2.0
  EXPECT_NEAR(grad_input_data[7], 2.0f, 1e-5f);

  // Position 13 (value 14) should get gradient 3.0
  EXPECT_NEAR(grad_input_data[13], 3.0f, 1e-5f);

  // Position 15 (value 16) should get gradient 4.0
  EXPECT_NEAR(grad_input_data[15], 4.0f, 1e-5f);
}

// Configuration Tests

TEST_F(LegacyMaxPool2DLayerTest, ComputeOutputShape) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_maxpool_shape");

  std::vector<size_t> input_shape = {2, 3, 8, 8};
  std::vector<size_t> expected_shape = {2, 3, 4, 4};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyMaxPool2DLayerTest, ComputeOutputShapeWithPadding) {
  LegacyMaxPool2DLayer layer(3, 3, 1, 1, 1, 1, "test_maxpool_shape_pad");

  std::vector<size_t> input_shape = {1, 1, 5, 5};
  std::vector<size_t> expected_shape = {1, 1, 5, 5};

  std::vector<size_t> output_shape = layer.compute_output_shape(input_shape);

  EXPECT_EQ(output_shape, expected_shape);
}

TEST_F(LegacyMaxPool2DLayerTest, GetConfig) {
  LegacyMaxPool2DLayer layer(3, 4, 2, 1, 1, 2, "test_maxpool_config");

  LayerConfig config = layer.get_config();

  EXPECT_EQ(config.name, "test_maxpool_config");
  EXPECT_EQ(config.get<size_t>("pool_h"), 3);
  EXPECT_EQ(config.get<size_t>("pool_w"), 4);
  EXPECT_EQ(config.get<size_t>("stride_h"), 2);
  EXPECT_EQ(config.get<size_t>("stride_w"), 1);
  EXPECT_EQ(config.get<size_t>("pad_h"), 1);
  EXPECT_EQ(config.get<size_t>("pad_w"), 2);
}

TEST_F(LegacyMaxPool2DLayerTest, CreateFromConfig) {
  LayerConfig config;
  config.name = "test_maxpool_recreate";
  config.parameters["pool_h"] = size_t(2);
  config.parameters["pool_w"] = size_t(2);
  config.parameters["stride_h"] = size_t(2);
  config.parameters["stride_w"] = size_t(2);
  config.parameters["pad_h"] = size_t(0);
  config.parameters["pad_w"] = size_t(0);

  auto layer = LegacyMaxPool2DLayer::create_from_config(config);

  EXPECT_NE(layer, nullptr);
  EXPECT_EQ(layer->type(), "maxpool2d");
}

TEST_F(LegacyMaxPool2DLayerTest, Clone) {
  LegacyMaxPool2DLayer original(3, 3, 1, 1, 1, 1, "test_maxpool_clone");

  auto cloned = original.clone();

  EXPECT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->type(), "maxpool2d");
  EXPECT_EQ(cloned->type(), original.type());
}

// Edge Cases

TEST_F(LegacyMaxPool2DLayerTest, EdgeCaseGlobalMaxPooling) {
  LegacyMaxPool2DLayer layer(4, 4, 1, 1, 0, 0, "test_global_maxpool");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  EXPECT_EQ(output_shape[2], 1);
  EXPECT_EQ(output_shape[3], 1);
  EXPECT_NEAR(output->data_as<float>()[0], 16.0f, 1e-5f);
}

TEST_F(LegacyMaxPool2DLayerTest, EdgeCaseZeroGradient) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_zero_gradient");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  input->fill(1.0f);

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), getCPU());
  gradient->fill(0.0f);

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);

  const float *grad_input_data = grad_input->data_as<float>();
  for (size_t i = 0; i < grad_input->size(); ++i) {
    EXPECT_NEAR(grad_input_data[i], 0.0f, 1e-5f);
  }
}

TEST_F(LegacyMaxPool2DLayerTest, EdgeCaseLargeValues) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_large_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1e6f * static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  verify_max_selection(input, output, 2, 2, 2, 2, 0, 0);
}

TEST_F(LegacyMaxPool2DLayerTest, EdgeCaseNegativeValues) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_negative_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = -static_cast<float>(17 - i);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  verify_max_selection(input, output, 2, 2, 2, 2, 0, 0);

  const float *output_data = output->data_as<float>();
  EXPECT_NEAR(output_data[0], -12.0f, 1e-5f);  // max of [-17,-16,-13,-12]
  EXPECT_NEAR(output_data[1], -10.0f, 1e-5f);  // max of [-15,-14,-11,-10]
  EXPECT_NEAR(output_data[2], -4.0f, 1e-5f);   // max of [-9,-8,-5,-4]
  EXPECT_NEAR(output_data[3], -2.0f, 1e-5f);   // max of [-7,-6,-3,-2]
}

TEST_F(LegacyMaxPool2DLayerTest, EdgeCaseMixedSignValues) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_mixed_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  verify_max_selection(input, output, 2, 2, 2, 2, 0, 0);
}

// Numerical Stability Tests

TEST_F(LegacyMaxPool2DLayerTest, NumericalStabilitySmallValues) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_small_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1e-6f * static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  verify_max_selection(input, output, 2, 2, 2, 2, 0, 0);
}

TEST_F(LegacyMaxPool2DLayerTest, BackwardNumericalStability) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_backward_stability");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data[i] = 1e-6f * static_cast<float>(i + 1);
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  Tensor gradient = Tensor::create<float>(output->shape(), getCPU());
  gradient->fill(1e-6f);

  Tensor grad_input = Tensor::create<float>(input->shape(), getCPU());
  layer.backward(gradient, grad_input);

  verify_gradient_shape(gradient, grad_input, input);
}

TEST_F(LegacyMaxPool2DLayerTest, NumericalStabilityExtremeValues) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_extreme_values");
  layer.set_device(getCPU());
  layer.init();

  Tensor input = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data = input->data_as<float>();
  input_data[0] = -1e10f;
  input_data[5] = 1e10f;
  for (int i = 1; i < 16; ++i) {
    if (i != 5) {
      input_data[i] = static_cast<float>(i);
    }
  }

  std::vector<size_t> output_shape = layer.compute_output_shape(input->shape());
  Tensor output = Tensor::create<float>(output_shape, getCPU());
  layer.forward(input, output);

  verify_output_shape(input, output, 2, 2, 2, 2, 0, 0);
  EXPECT_NEAR(output->data_as<float>()[0], 1e10f, 1e5f);
}

TEST_F(LegacyMaxPool2DLayerTest, MultipleForwardBackwardPasses) {
  LegacyMaxPool2DLayer layer(2, 2, 2, 2, 0, 0, "test_multiple_passes");
  layer.set_device(getCPU());
  layer.init();

  // First pass
  Tensor input1 = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  input1->fill(1.0f);
  std::vector<size_t> output_shape1 = layer.compute_output_shape(input1->shape());
  Tensor output1 = Tensor::create<float>(output_shape1, getCPU());
  layer.forward(input1, output1);
  Tensor gradient1 = Tensor::create<float>(output1->shape(), getCPU());
  gradient1->fill(1.0f);
  Tensor grad_input1 = Tensor::create<float>(input1->shape(), getCPU());
  layer.backward(gradient1, grad_input1);

  // Second pass
  Tensor input2 = Tensor::create<float>({1, 1, 4, 4}, getCPU());
  float *input_data2 = input2->data_as<float>();
  for (int i = 0; i < 16; ++i) {
    input_data2[i] = static_cast<float>(i + 1);
  }
  std::vector<size_t> output_shape2 = layer.compute_output_shape(input2->shape());
  Tensor output2 = Tensor::create<float>(output_shape2, getCPU());
  layer.forward(input2, output2);
  Tensor gradient2 = Tensor::create<float>(output2->shape(), getCPU());
  gradient2->fill(1.0f);
  Tensor grad_input2 = Tensor::create<float>(input2->shape(), getCPU());
  layer.backward(gradient2, grad_input2);

  verify_gradient_shape(gradient2, grad_input2, input2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
