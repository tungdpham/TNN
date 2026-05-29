/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "nn/layers_impl/legacy_maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"
#include "test_graph_utils.hpp"

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
    Vec<std::string> device_ids = manager.getAvailableDeviceIDs();

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

  void compareTensors(const ConstTensor &expected, const ConstTensor &actual,
                      float tolerance = 1e-3f, const std::string &context = "") {
    ASSERT_EQ(expected->shape(), actual->shape()) << context << " Tensors have different shapes";

    Tensor expected_cpu = expected->device().device_type() == DeviceType::CPU
                              ? expected->clone()
                              : expected->to_device(getHost());
    Tensor actual_cpu = actual->device().device_type() == DeviceType::CPU
                            ? actual->clone()
                            : actual->to_device(getHost());

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

  auto cpu_layer_layer = LegacyConv2DLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, true, "cpu_conv");
  LegacyConv2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer = LegacyConv2DLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, true, "gpu_conv");
  LegacyConv2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  *gpu_layer->parameters()[0] = *cpu_layer->parameters()[0]->to_device(getGPU());
  if (cpu_layer->parameters().size() > 1) {
    *gpu_layer->parameters()[1] = *cpu_layer->parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getHost());
  input->fill_random_uniform(2.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-3f, "LegacyConv2DLayerImpl Forward:");
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

  auto cpu_layer_layer = LegacyConv2DLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, true, "cpu_conv");
  LegacyConv2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer = LegacyConv2DLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, true, "gpu_conv");
  LegacyConv2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  *gpu_layer->parameters()[0] = *cpu_layer->parameters()[0]->to_device(getGPU());
  if (cpu_layer->parameters().size() > 1) {
    *gpu_layer->parameters()[1] = *cpu_layer->parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getHost());
  input->fill_random_uniform(2.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  Tensor grad_output = make_tensor<float>({batch_size, out_channels, input_h, input_w}, getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f,
                 "LegacyConv2DLayerImpl Backward Input Gradient:");

  compareTensors(cpu_layer->gradients()[0], gpu_layer->gradients()[0], 1e-2f,
                 "LegacyConv2DLayerImpl Backward Weight Gradient:");

  if (cpu_layer->gradients().size() > 1) {
    compareTensors(cpu_layer->gradients()[1], gpu_layer->gradients()[1], 1e-2f,
                   "LegacyConv2DLayerImpl Backward Bias Gradient:");
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

  auto cpu_layer_layer = LegacyConv2DLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, false, "cpu_conv_strided");
  LegacyConv2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer = LegacyConv2DLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, false, "gpu_conv_strided");
  LegacyConv2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  *gpu_layer->parameters()[0] = *cpu_layer->parameters()[0]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getHost());
  input->fill_random_uniform(1.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-3f, "LegacyConv2DLayerImpl Strided Forward:");

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getHost());
  grad_output->fill_random_uniform(1.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];
  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "LegacyConv2DLayerImpl Strided Backward:");
  compareTensors(cpu_layer->gradients()[0], gpu_layer->gradients()[0], 1e-2f,
                 "LegacyConv2DLayerImpl Strided Weight Gradient:");
}

TEST_F(LayerIntegrationTest, DenseLayerForwardBasic) {
  const size_t batch_size = 4;
  const size_t input_features = 128;
  const size_t output_features = 64;

  auto cpu_layer_layer = DenseLayer(input_features, output_features, true, "cpu_dense");
  DenseLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer = DenseLayer(input_features, output_features, true, "gpu_dense");
  DenseLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  *gpu_layer->parameters()[0] = *cpu_layer->parameters()[0]->to_device(getGPU());
  if (cpu_layer->parameters().size() > 1) {
    *gpu_layer->parameters()[1] = *cpu_layer->parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, input_features}, getHost());
  input->fill_random_uniform(2.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-3f, "DenseLayerImpl Forward:");
}

TEST_F(LayerIntegrationTest, DenseLayerBackwardBasic) {
  const size_t batch_size = 4;
  const size_t input_features = 128;
  const size_t output_features = 64;

  auto cpu_layer_layer = DenseLayer(input_features, output_features, true, "cpu_dense");
  DenseLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer = DenseLayer(input_features, output_features, true, "gpu_dense");
  DenseLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  *gpu_layer->parameters()[0] = *cpu_layer->parameters()[0]->to_device(getGPU());
  if (cpu_layer->parameters().size() > 1) {
    *gpu_layer->parameters()[1] = *cpu_layer->parameters()[1]->to_device(getGPU());
  }

  Tensor input = make_tensor<float>({batch_size, input_features}, getHost());
  input->fill_random_uniform(2.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  Tensor grad_output = make_tensor<float>({batch_size, output_features}, getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "DenseLayerImpl Backward Input Gradient:");
  compareTensors(cpu_layer->gradients()[0], gpu_layer->gradients()[0], 1e-2f,
                 "DenseLayerImpl Backward Weight Gradient:");

  if (cpu_layer->gradients().size() > 1) {
    compareTensors(cpu_layer->gradients()[1], gpu_layer->gradients()[1], 1e-2f,
                   "DenseLayerImpl Backward Bias Gradient:");
  }
}

TEST_F(LayerIntegrationTest, DenseLayerLargeMatrix) {
  const size_t batch_size = 8;
  const size_t input_features = 512;
  const size_t output_features = 256;

  auto cpu_layer_layer = DenseLayer(input_features, output_features, false, "cpu_dense_large");
  DenseLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer = DenseLayer(input_features, output_features, false, "gpu_dense_large");
  DenseLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  *gpu_layer->parameters()[0] = *cpu_layer->parameters()[0]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, input_features}, getHost());
  input->fill_random_uniform(1.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-3f, "DenseLayerImpl Large Forward:");

  Tensor grad_output = make_tensor<float>({batch_size, output_features}, getHost());
  grad_output->fill_random_uniform(1.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];
  compareTensors(cpu_grad_input, gpu_grad_input, 1e-2f, "DenseLayerImpl Large Backward:");
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

  auto cpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool");
  LegacyMaxPool2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool");
  LegacyMaxPool2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getHost());
  input->fill_random_uniform(10.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-4f, "LegacyMaxPool2DLayerImpl Forward:");
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

  auto cpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool");
  LegacyMaxPool2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool");
  LegacyMaxPool2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getHost());
  input->fill_random_uniform(20.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f, "LegacyMaxPool2DLayerImpl Backward:");
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

  auto cpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool_pad");
  LegacyMaxPool2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool_pad");
  LegacyMaxPool2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getHost());
  input->fill_random_uniform(10.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-4f, "LegacyMaxPool2DLayerImpl Padded Forward:");

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f,
                 "LegacyMaxPool2DLayerImpl Padded Backward:");
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

  auto cpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "cpu_maxpool_nonsq");
  LegacyMaxPool2DLayerImpl *cpu_layer = cpu_layer_layer.get();
  auto gpu_layer_layer =
      LegacyMaxPool2DLayer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, "gpu_maxpool_nonsq");
  LegacyMaxPool2DLayerImpl *gpu_layer = gpu_layer_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_single_layer(cpu_layer_layer, cpu_allocator);

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_single_layer(gpu_layer_layer, gpu_allocator);

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, getHost());
  input->fill_random_uniform(16.0f);

  auto cpu_output = cpu_layer->forward({input})[0];
  auto gpu_output = gpu_layer->forward({input})[0];

  compareTensors(cpu_output, gpu_output, 1e-4f, "LegacyMaxPool2DLayerImpl Non-square Forward:");

  Tensor grad_output = make_tensor<float>(cpu_output->shape(), getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_input = cpu_layer->backward({grad_output})[0];
  auto gpu_grad_input = gpu_layer->backward({grad_output})[0];

  compareTensors(cpu_grad_input, gpu_grad_input, 1e-4f,
                 "LegacyMaxPool2DLayerImpl Non-square Backward:");
}

TEST_F(LayerIntegrationTest, Conv2DMaxPoolPipeline) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 16;
  const size_t input_w = 16;

  auto cpu_conv_layer =
      LegacyConv2DLayer(in_channels, out_channels, 3, 3, 1, 1, 1, 1, true, "cpu_conv");
  LegacyConv2DLayerImpl *cpu_conv = cpu_conv_layer.get();
  auto cpu_pool_layer = LegacyMaxPool2DLayer(2, 2, 2, 2, 0, 0, "cpu_pool");
  LegacyMaxPool2DLayerImpl *cpu_pool = cpu_pool_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_two_layer_chain(cpu_conv_layer, cpu_pool_layer, cpu_allocator);

  auto gpu_conv_layer =
      LegacyConv2DLayer(in_channels, out_channels, 3, 3, 1, 1, 1, 1, true, "gpu_conv");
  LegacyConv2DLayerImpl *gpu_conv = gpu_conv_layer.get();
  auto gpu_pool_layer = LegacyMaxPool2DLayer(2, 2, 2, 2, 0, 0, "gpu_pool");
  LegacyMaxPool2DLayerImpl *gpu_pool = gpu_pool_layer.get();

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_two_layer_chain(gpu_conv_layer, gpu_pool_layer, gpu_allocator);

  *gpu_conv->parameters()[0] = *cpu_conv->parameters()[0]->to_device(getGPU());
  *gpu_conv->parameters()[1] = *cpu_conv->parameters()[1]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getHost());
  input->fill_random_uniform(2.0f);

  auto cpu_conv_out = cpu_conv->forward({input})[0];
  auto cpu_pool_out = cpu_pool->forward({cpu_conv_out})[0];

  auto gpu_conv_out = gpu_conv->forward({input})[0];
  auto gpu_pool_out = gpu_pool->forward({gpu_conv_out})[0];

  compareTensors(cpu_pool_out, gpu_pool_out, 1e-3f, "Conv2D-MaxPool Pipeline Forward:");

  Tensor grad_output = make_tensor<float>(cpu_pool_out->shape(), getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_pool = cpu_pool->backward({grad_output})[0];
  auto cpu_grad_conv = cpu_conv->backward({cpu_grad_pool})[0];

  auto gpu_grad_pool = gpu_pool->backward({grad_output})[0];
  auto gpu_grad_conv = gpu_conv->backward({gpu_grad_pool})[0];

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

  auto cpu_conv_layer =
      LegacyConv2DLayer(in_channels, out_channels, 3, 3, 1, 1, 1, 1, false, "cpu_conv");
  LegacyConv2DLayerImpl *cpu_conv = cpu_conv_layer.get();
  auto cpu_dense_layer = DenseLayer(flattened_size, dense_output, true, "cpu_dense");
  DenseLayerImpl *cpu_dense = cpu_dense_layer.get();

  auto &cpu_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph cpu_graph = test::compile_two_layer_chain(cpu_conv_layer, cpu_dense_layer, cpu_allocator);

  auto gpu_conv_layer =
      LegacyConv2DLayer(in_channels, out_channels, 3, 3, 1, 1, 1, 1, false, "gpu_conv");
  LegacyConv2DLayerImpl *gpu_conv = gpu_conv_layer.get();
  auto gpu_dense_layer = DenseLayer(flattened_size, dense_output, true, "gpu_dense");
  DenseLayerImpl *gpu_dense = gpu_dense_layer.get();

  auto &gpu_allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  Graph gpu_graph = test::compile_two_layer_chain(gpu_conv_layer, gpu_dense_layer, gpu_allocator);

  *gpu_conv->parameters()[0] = *cpu_conv->parameters()[0]->to_device(getGPU());
  *gpu_dense->parameters()[0] = *cpu_dense->parameters()[0]->to_device(getGPU());
  *gpu_dense->parameters()[1] = *cpu_dense->parameters()[1]->to_device(getGPU());

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, getHost());
  input->fill_random_uniform(2.0f);

  auto cpu_conv_out = cpu_conv->forward({input})[0];

  Tensor cpu_conv_flat = make_tensor<float>({batch_size, flattened_size}, getHost());
  cpu_conv_out->copy_to(cpu_conv_flat);
  auto cpu_dense_out = cpu_dense->forward({cpu_conv_flat})[0];

  auto gpu_conv_out = gpu_conv->forward({input})[0];
  Tensor gpu_conv_flat = make_tensor<float>({batch_size, flattened_size}, getGPU());
  gpu_conv_out->copy_to(gpu_conv_flat);
  auto gpu_dense_out = gpu_dense->forward({gpu_conv_flat})[0];

  compareTensors(cpu_dense_out, gpu_dense_out, 1e-3f, "Conv2D-Dense Pipeline Forward:");

  Tensor grad_output = make_tensor<float>({batch_size, dense_output}, getHost());
  grad_output->fill_random_uniform(2.0f);

  auto cpu_grad_dense = cpu_dense->backward({grad_output})[0];
  Tensor cpu_grad_dense_reshape =
      make_tensor<float>({batch_size, out_channels, input_h, input_w}, getHost());
  cpu_grad_dense->copy_to(cpu_grad_dense_reshape);
  auto cpu_grad_conv = cpu_conv->backward({cpu_grad_dense_reshape})[0];

  auto gpu_grad_dense = gpu_dense->backward({grad_output})[0];
  Tensor gpu_grad_dense_reshape =
      make_tensor<float>({batch_size, out_channels, input_h, input_w}, getGPU());
  gpu_grad_dense->copy_to(gpu_grad_dense_reshape);
  auto gpu_grad_conv = gpu_conv->backward({gpu_grad_dense_reshape})[0];

  compareTensors(cpu_grad_conv, gpu_grad_conv, 1e-2f, "Conv2D-Dense Pipeline Backward:");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
