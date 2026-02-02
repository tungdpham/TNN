/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "device/device_manager.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "nn/layers_impl/legacy_maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

#ifdef USE_CUDA

/**
 * Integration test fixture for testing complete layer implementations
 * comparing CPU vs GPU device execution.
 */
class LayerIntegrationTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    has_gpu_ = false;

    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.device_type() == DeviceType::CPU) {
        has_cpu_ = true;
      } else if (device.device_type() == DeviceType::GPU) {
        has_gpu_ = true;
      }
    }

    if (!has_cpu_) {
      GTEST_SKIP() << "No CPU device available";
    }
    if (!has_gpu_) {
      GTEST_SKIP() << "No GPU device available, skipping layer integration tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  void compareTensors(const Tensor &expected, const Tensor &actual, float tolerance = 1e-3f,
                      const std::string &context = "") {
    ASSERT_EQ(expected->shape(), actual->shape()) << context << " Tensors have different shapes";

    Tensor expected_cpu = expected->device().device_type() == DeviceType::CPU
                              ? expected->clone()
                              : expected->to_device(getCPU());
    Tensor actual_cpu = actual->device().device_type() == DeviceType::CPU
                            ? actual->clone()
                            : actual->to_device(getCPU());

    size_t total_elements = expected_cpu->size();
    const float *expected_data = expected_cpu->data_as<float>();
    const float *actual_data = actual_cpu->data_as<float>();

    size_t mismatch_count = 0;
    const size_t max_mismatches_to_show = 10;

    for (size_t i = 0; i < total_elements; ++i) {
      if (std::abs(expected_data[i] - actual_data[i]) > tolerance) {
        if (mismatch_count < max_mismatches_to_show) {
          std::cerr << context << " Mismatch at index " << i << ": Expected " << expected_data[i]
                    << ", Got " << actual_data[i]
                    << ", Diff: " << std::abs(expected_data[i] - actual_data[i]) << std::endl;
        }
        mismatch_count++;
      }
    }

    EXPECT_EQ(0, mismatch_count) << context << " Found " << mismatch_count << " mismatches out of "
                                 << total_elements << " elements";
  }

  bool has_cpu_;
  bool has_gpu_;
};

TEST_F(LayerIntegrationTest, LegacyConv2DLayerForwardBasic) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 48;
  const size_t input_h = 28;
  const size_t input_w = 28;
  const size_t kernel_h = 3;
  const size_t kernel_w = 3;
  const size_t stride_h = 1;
  const size_t stride_w = 1;
  const size_t pad_h = 1;
  const size_t pad_w = 1;

  LegacyConv2DLayer cpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                              pad_h, pad_w, true, "cpu_conv");
  LegacyConv2DLayer gpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                              pad_h, pad_w, true, "gpu_conv");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  *gpu_layer.parameters()[0] = *cpu_layer.parameters()[0]->to_device(getGPU());
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = *cpu_layer.parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(2.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);

  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-3f, "LegacyConv2DLayer Forward:");
}

TEST_F(LayerIntegrationTest, LegacyConv2DLayerBackwardBasic) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 4;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t kernel_h = 3;
  const size_t kernel_w = 3;
  const size_t stride_h = 1;
  const size_t stride_w = 1;
  const size_t pad_h = 1;
  const size_t pad_w = 1;

  LegacyConv2DLayer cpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                              pad_h, pad_w, true, "cpu_conv");
  LegacyConv2DLayer gpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                              pad_h, pad_w, true, "gpu_conv");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  *gpu_layer.parameters()[0] = *cpu_layer.parameters()[0]->to_device(getGPU());
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = *cpu_layer.parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(2.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  Tensor grad_output = make_tensor<float>({batch_size, out_channels, input_h, input_w}, getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);

  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f,
                 "LegacyConv2DLayer Backward Input Gradient:");

  compareTensors(cpu_layer.gradients()[0], gpu_layer.gradients()[0], 1e-2f,
                 "LegacyConv2DLayer Backward Weight Gradient:");

  if (cpu_layer.gradients().size() > 1) {
    compareTensors(cpu_layer.gradients()[1], gpu_layer.gradients()[1], 1e-2f,
                   "LegacyConv2DLayer Backward Bias Gradient:");
  }
}

TEST_F(LayerIntegrationTest, LegacyConv2DLayerStridedConvolution) {
  const size_t batch_size = 1;
  const size_t in_channels = 2;
  const size_t out_channels = 3;
  const size_t input_h = 16;
  const size_t input_w = 16;
  const size_t kernel_h = 5;
  const size_t kernel_w = 5;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t pad_h = 2;
  const size_t pad_w = 2;

  LegacyConv2DLayer cpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                              pad_h, pad_w, false, "cpu_conv_strided");
  LegacyConv2DLayer gpu_layer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                              pad_h, pad_w, false, "gpu_conv_strided");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  *gpu_layer.parameters()[0] = *cpu_layer.parameters()[0]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-3f, "LegacyConv2DLayer Strided Forward:");

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getCPU());
  grad_output->fill_random_uniform(1.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);
  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);
  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "LegacyConv2DLayer Strided Backward:");
  compareTensors(cpu_layer.gradients()[0], gpu_layer.gradients()[0], 1e-2f,
                 "LegacyConv2DLayer Strided Weight Gradient:");
}

TEST_F(LayerIntegrationTest, DenseLayerForwardBasic) {
  const size_t batch_size = 4;
  const size_t input_features = 128;
  const size_t output_features = 64;

  DenseLayer cpu_layer(input_features, output_features, true, "cpu_dense");
  DenseLayer gpu_layer(input_features, output_features, true, "gpu_dense");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  *gpu_layer.parameters()[0] = *cpu_layer.parameters()[0]->to_device(getGPU());
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = *cpu_layer.parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, input_features}, getCPU());
  input->fill_random_uniform(2.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-3f, "DenseLayer Forward:");
}

TEST_F(LayerIntegrationTest, DenseLayerBackwardBasic) {
  const size_t batch_size = 4;
  const size_t input_features = 128;
  const size_t output_features = 64;

  DenseLayer cpu_layer(input_features, output_features, true, "cpu_dense");
  DenseLayer gpu_layer(input_features, output_features, true, "gpu_dense");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  *gpu_layer.parameters()[0] = *cpu_layer.parameters()[0]->to_device(getGPU());
  if (cpu_layer.parameters().size() > 1) {
    *gpu_layer.parameters()[1] = *cpu_layer.parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, input_features}, getCPU());
  input->fill_random_uniform(2.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  Tensor grad_output = make_tensor<float>({batch_size, output_features}, getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);
  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "DenseLayer Backward Input Gradient:");
  compareTensors(cpu_layer.gradients()[0], gpu_layer.gradients()[0], 1e-2f,
                 "DenseLayer Backward Weight Gradient:");

  if (cpu_layer.gradients().size() > 1) {
    compareTensors(cpu_layer.gradients()[1], gpu_layer.gradients()[1], 1e-2f,
                   "DenseLayer Backward Bias Gradient:");
  }
}

TEST_F(LayerIntegrationTest, DenseLayerLargeMatrix) {
  const size_t batch_size = 8;
  const size_t input_features = 512;
  const size_t output_features = 256;

  DenseLayer cpu_layer(input_features, output_features, false, "cpu_dense_large");
  DenseLayer gpu_layer(input_features, output_features, false, "gpu_dense_large");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  *gpu_layer.parameters()[0] = *cpu_layer.parameters()[0]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, input_features}, getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-3f, "DenseLayer Large Forward:");

  Tensor grad_output = make_tensor<float>({batch_size, output_features}, getCPU());
  grad_output->fill_random_uniform(1.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);
  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);
  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "DenseLayer Large Backward:");
}

TEST_F(LayerIntegrationTest, LegacyMaxPool2DLayerForwardBasic) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t pad_h = 0;
  const size_t pad_w = 0;

  LegacyMaxPool2DLayer cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool");
  LegacyMaxPool2DLayer gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(10.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-4f, "LegacyMaxPool2DLayer Forward:");
}

TEST_F(LayerIntegrationTest, LegacyMaxPool2DLayerBackwardBasic) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t pad_h = 0;
  const size_t pad_w = 0;

  LegacyMaxPool2DLayer cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool");
  LegacyMaxPool2DLayer gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(20.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);
  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f, "LegacyMaxPool2DLayer Backward:");
}

TEST_F(LayerIntegrationTest, LegacyMaxPool2DLayerWithPadding) {
  const size_t batch_size = 1;
  const size_t channels = 4;
  const size_t input_h = 7;
  const size_t input_w = 7;
  const size_t pool_h = 3;
  const size_t pool_w = 3;
  const size_t stride_h = 1;
  const size_t stride_w = 1;
  const size_t pad_h = 1;
  const size_t pad_w = 1;

  LegacyMaxPool2DLayer cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                 "cpu_maxpool_pad");
  LegacyMaxPool2DLayer gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                 "gpu_maxpool_pad");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(10.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-4f, "LegacyMaxPool2DLayer Padded Forward:");

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);
  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f, "LegacyMaxPool2DLayer Padded Backward:");
}

TEST_F(LayerIntegrationTest, LegacyMaxPool2DLayerNonSquare) {
  const size_t batch_size = 3;
  const size_t channels = 2;
  const size_t input_h = 12;
  const size_t input_w = 16;
  const size_t pool_h = 3;
  const size_t pool_w = 4;
  const size_t stride_h = 3;
  const size_t stride_w = 4;
  const size_t pad_h = 0;
  const size_t pad_w = 0;

  LegacyMaxPool2DLayer cpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                 "cpu_maxpool_nonsq");
  LegacyMaxPool2DLayer gpu_layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                 "gpu_maxpool_nonsq");

  cpu_layer.set_device(getCPU());
  gpu_layer.set_device(getGPU());

  cpu_layer.init();
  gpu_layer.init();

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(16.0f);

  std::vector<size_t> output_shape = cpu_layer.compute_output_shape(input->shape());
  Tensor cpu_output = make_tensor<float>(output_shape, getCPU());
  cpu_layer.forward(input, cpu_output);
  Tensor gpu_output = make_tensor<float>(output_shape, getGPU());
  gpu_layer.forward(input, gpu_output);

  compareTensors(cpu_output, gpu_output, 1e-4f, "LegacyMaxPool2DLayer Non-square Forward:");

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_input = make_tensor<float>(input->shape(), getCPU());
  cpu_layer.backward(grad_output, cpu_grad_input);
  Tensor gpu_grad_input = make_tensor<float>(input->shape(), getGPU());
  gpu_layer.backward(grad_output, gpu_grad_input);

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f,
                 "LegacyMaxPool2DLayer Non-square Backward:");
}

TEST_F(LayerIntegrationTest, Conv2DMaxPoolPipeline) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 16;
  const size_t input_w = 16;

  LegacyConv2DLayer cpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, true, "cpu_conv");
  LegacyMaxPool2DLayer cpu_pool(2, 2, 2, 2, 0, 0, "cpu_pool");

  cpu_conv.set_device(getCPU());
  cpu_pool.set_device(getCPU());
  cpu_conv.init();
  cpu_pool.init();

  LegacyConv2DLayer gpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, true, "gpu_conv");
  LegacyMaxPool2DLayer gpu_pool(2, 2, 2, 2, 0, 0, "gpu_pool");

  gpu_conv.set_device(getGPU());
  gpu_pool.set_device(getGPU());
  gpu_conv.init();
  gpu_pool.init();

  *gpu_conv.parameters()[0] = *cpu_conv.parameters()[0]->to_device(getGPU());
  *gpu_conv.parameters()[1] = *cpu_conv.parameters()[1]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(2.0f);

  std::vector<size_t> conv_output_shape = cpu_conv.compute_output_shape(input->shape());
  Tensor cpu_conv_out = make_tensor<float>(conv_output_shape, getCPU());
  cpu_conv.forward(input, cpu_conv_out);
  std::vector<size_t> pool_output_shape = cpu_pool.compute_output_shape(cpu_conv_out->shape());
  Tensor cpu_pool_out = make_tensor<float>(pool_output_shape, getCPU());
  cpu_pool.forward(cpu_conv_out, cpu_pool_out);

  Tensor gpu_conv_out = make_tensor<float>(conv_output_shape, getGPU());
  gpu_conv.forward(input, gpu_conv_out);
  Tensor gpu_pool_out = make_tensor<float>(pool_output_shape, getGPU());
  gpu_pool.forward(gpu_conv_out, gpu_pool_out);

  compareTensors(cpu_pool_out, gpu_pool_out, 1e-3f, "Conv2D-MaxPool Pipeline Forward:");

  Tensor grad_output = make_tensor<float>(cpu_pool_out->shape(), getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_pool = make_tensor<float>(cpu_conv_out->shape(), getCPU());
  cpu_pool.backward(grad_output, cpu_grad_pool);
  Tensor cpu_grad_conv = make_tensor<float>(input->shape(), getCPU());
  cpu_conv.backward(cpu_grad_pool, cpu_grad_conv);

  Tensor gpu_grad_pool = make_tensor<float>(gpu_conv_out->shape(), getGPU());
  gpu_pool.backward(grad_output, gpu_grad_pool);
  Tensor gpu_grad_conv = make_tensor<float>(input->shape(), getGPU());
  gpu_conv.backward(gpu_grad_pool, gpu_grad_conv);

  compareTensors(cpu_grad_conv, gpu_grad_conv, 1e-2f, "Conv2D-MaxPool Pipeline Backward:");
}

TEST_F(LayerIntegrationTest, Conv2DDensePipeline) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t dense_output = 10;

  const size_t flattened_size = input_h * input_w * out_channels;

  LegacyConv2DLayer cpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, false, "cpu_conv");
  DenseLayer cpu_dense(flattened_size, dense_output, true, "cpu_dense");

  cpu_conv.set_device(getCPU());
  cpu_dense.set_device(getCPU());
  cpu_conv.init();
  cpu_dense.init();

  LegacyConv2DLayer gpu_conv(in_channels, out_channels, 3, 3, 1, 1, 1, 1, false, "gpu_conv");
  DenseLayer gpu_dense(flattened_size, dense_output, true, "gpu_dense");

  gpu_conv.set_device(getGPU());
  gpu_dense.set_device(getGPU());
  gpu_conv.init();
  gpu_dense.init();

  *gpu_conv.parameters()[0] = *cpu_conv.parameters()[0]->to_device(getGPU());
  *gpu_dense.parameters()[0] = *cpu_dense.parameters()[0]->to_device(getGPU());
  *gpu_dense.parameters()[1] = *cpu_dense.parameters()[1]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getCPU());
  input->fill_random_uniform(2.0f);

  std::vector<size_t> conv_output_shape = cpu_conv.compute_output_shape(input->shape());
  Tensor cpu_conv_out = make_tensor<float>(conv_output_shape, getCPU());
  cpu_conv.forward(input, cpu_conv_out);

  Tensor cpu_conv_flat = make_tensor<float>({batch_size, flattened_size}, getCPU());
  std::vector<size_t> dense_output_shape = cpu_dense.compute_output_shape(cpu_conv_flat->shape());
  Tensor cpu_dense_out = make_tensor<float>(dense_output_shape, getCPU());
  cpu_conv_out->copy_to(cpu_conv_flat);
  cpu_dense.forward(cpu_conv_flat, cpu_dense_out);

  Tensor gpu_conv_out = make_tensor<float>(conv_output_shape, getGPU());
  gpu_conv.forward(input, gpu_conv_out);
  Tensor gpu_conv_flat = make_tensor<float>({batch_size, flattened_size}, getGPU());
  Tensor gpu_dense_out = make_tensor<float>(dense_output_shape, getGPU());
  gpu_conv_out->copy_to(gpu_conv_flat);
  gpu_dense.forward(gpu_conv_flat, gpu_dense_out);

  compareTensors(cpu_dense_out, gpu_dense_out, 1e-3f, "Conv2D-Dense Pipeline Forward:");

  Tensor grad_output = make_tensor<float>({batch_size, dense_output}, getCPU());
  grad_output->fill_random_uniform(2.0f);

  Tensor cpu_grad_dense = make_tensor<float>(cpu_conv_flat->shape(), getCPU());
  cpu_dense.backward(grad_output, cpu_grad_dense);
  Tensor cpu_grad_dense_reshape =
      make_tensor<float>({batch_size, out_channels, input_h, input_w}, getCPU());
  cpu_grad_dense->copy_to(cpu_grad_dense_reshape);
  Tensor cpu_grad_conv = make_tensor<float>(input->shape(), getCPU());
  cpu_conv.backward(cpu_grad_dense_reshape, cpu_grad_conv);

  Tensor gpu_grad_dense = make_tensor<float>(gpu_conv_flat->shape(), getGPU());
  gpu_dense.backward(grad_output, gpu_grad_dense);
  Tensor gpu_grad_dense_reshape =
      make_tensor<float>({batch_size, out_channels, input_h, input_w}, getGPU());
  gpu_grad_dense->copy_to(gpu_grad_dense_reshape);
  Tensor gpu_grad_conv = make_tensor<float>(input->shape(), getGPU());
  gpu_conv.backward(gpu_grad_dense_reshape, gpu_grad_conv);

  compareTensors(cpu_grad_conv, gpu_grad_conv, 1e-2f, "Conv2D-Dense Pipeline Backward:");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
