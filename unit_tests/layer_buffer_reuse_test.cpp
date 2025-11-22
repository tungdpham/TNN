/*
 * Test to verify that layers properly handle buffer reuse
 * across multiple forward/backward passes
 */

#include <gtest/gtest.h>
#include <memory>

#include "nn/layers_impl/activation_layer.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/flatten_layer.hpp"
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

// Test that Conv2D produces consistent outputs when buffer is reused
TEST(LayerBufferReuseTest, Conv2DConsistentOutput) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 4;
  const size_t input_h = 8;
  const size_t input_w = 8;

  auto layer = std::make_unique<Conv2DLayer<float>>(in_channels, out_channels, 3, 3, 1, 1, 1, 1,
                                                    true, "test_conv");
  layer->set_device(&getCPU());
  layer->initialize();

  // Create input
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, &getCPU());
  input.fill_random_uniform(1.0f);

  // First forward pass (micro_batch_id = 0)
  const auto &output1 = layer->forward(input, 0);
  Tensor<float> output1_copy = output1.clone();

  // Second forward pass with same input and same micro_batch_id
  // This should reuse the buffer and produce identical output
  const auto &output2 = layer->forward(input, 0);

  // Verify outputs are identical
  ASSERT_EQ(output1_copy.shape(), output2.shape());
  for (size_t i = 0; i < output1_copy.size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy.data_ptr().get()[i], output2.data_ptr().get()[i])
        << "Mismatch at index " << i;
  }
}

// Test that Dense layer produces consistent outputs when buffer is reused
TEST(LayerBufferReuseTest, DenseConsistentOutput) {
  const size_t batch_size = 4;
  const size_t input_features = 32;
  const size_t output_features = 10;

  auto layer =
      std::make_unique<DenseLayer<float>>(input_features, output_features, true, "test_dense");
  layer->set_device(&getCPU());
  layer->initialize();

  // Create input
  Tensor<float> input({batch_size, input_features, 1, 1}, &getCPU());
  input.fill_random_uniform(1.0f);

  // First forward pass (micro_batch_id = 0)
  const auto &output1 = layer->forward(input, 0);
  Tensor<float> output1_copy = output1.clone();

  // Second forward pass with same input and same micro_batch_id
  const auto &output2 = layer->forward(input, 0);

  // Verify outputs are identical
  ASSERT_EQ(output1_copy.shape(), output2.shape());
  for (size_t i = 0; i < output1_copy.size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy.data_ptr().get()[i], output2.data_ptr().get()[i])
        << "Mismatch at index " << i << ": " << output1_copy.data_ptr().get()[i] << " vs "
        << output2.data_ptr().get()[i];
  }
}

// Test that MaxPool2D produces consistent outputs when buffer is reused
TEST(LayerBufferReuseTest, MaxPool2DConsistentOutput) {
  const size_t batch_size = 2;
  const size_t channels = 8;
  const size_t input_h = 8;
  const size_t input_w = 8;

  auto layer = std::make_unique<MaxPool2DLayer<float>>(2, 2, 2, 2, 0, 0, "test_pool");
  layer->set_device(&getCPU());

  // Create input
  Tensor<float> input({batch_size, channels, input_h, input_w}, &getCPU());
  input.fill_random_uniform(1.0f);

  // First forward pass (micro_batch_id = 0)
  const auto &output1 = layer->forward(input, 0);
  Tensor<float> output1_copy = output1.clone();

  // Second forward pass with same input and same micro_batch_id
  const auto &output2 = layer->forward(input, 0);

  // Verify outputs are identical
  ASSERT_EQ(output1_copy.shape(), output2.shape());
  for (size_t i = 0; i < output1_copy.size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy.data_ptr().get()[i], output2.data_ptr().get()[i])
        << "Mismatch at index " << i;
  }
}

// Test that Activation layer produces consistent outputs when buffer is reused
TEST(LayerBufferReuseTest, ActivationConsistentOutput) {
  const size_t batch_size = 2;
  const size_t channels = 8;
  const size_t h = 4;
  const size_t w = 4;

  auto factory = ActivationFactory<float>();
  factory.register_defaults();
  auto activation = factory.create("relu");
  auto layer = std::make_unique<ActivationLayer<float>>(std::move(activation), "test_relu");
  layer->set_device(&getCPU());

  // Create input
  Tensor<float> input({batch_size, channels, h, w}, &getCPU());
  input.fill_random_uniform(1.0f);

  // First forward pass (micro_batch_id = 0)
  const auto &output1 = layer->forward(input, 0);
  Tensor<float> output1_copy = output1.clone();

  // Second forward pass with same input and same micro_batch_id
  const auto &output2 = layer->forward(input, 0);

  // Verify outputs are identical
  ASSERT_EQ(output1_copy.shape(), output2.shape());
  for (size_t i = 0; i < output1_copy.size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy.data_ptr().get()[i], output2.data_ptr().get()[i])
        << "Mismatch at index " << i;
  }
}

// Test that Flatten layer produces consistent outputs when buffer is reused
TEST(LayerBufferReuseTest, FlattenConsistentOutput) {
  const size_t batch_size = 2;
  const size_t channels = 8;
  const size_t h = 4;
  const size_t w = 4;

  auto layer = std::make_unique<FlattenLayer<float>>("test_flatten");
  layer->set_device(&getCPU());

  // Create input
  Tensor<float> input({batch_size, channels, h, w}, &getCPU());
  input.fill_random_uniform(1.0f);

  // First forward pass (micro_batch_id = 0)
  const auto &output1 = layer->forward(input, 0);
  Tensor<float> output1_copy = output1.clone();

  // Second forward pass with same input and same micro_batch_id
  const auto &output2 = layer->forward(input, 0);

  // Verify outputs are identical
  ASSERT_EQ(output1_copy.shape(), output2.shape());
  for (size_t i = 0; i < output1_copy.size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy.data_ptr().get()[i], output2.data_ptr().get()[i])
        << "Mismatch at index " << i;
  }
}

// Test multiple passes to simulate epoch behavior
TEST(LayerBufferReuseTest, DenseMultipleEpochs) {
  const size_t batch_size = 4;
  const size_t input_features = 16;
  const size_t output_features = 8;

  auto layer =
      std::make_unique<DenseLayer<float>>(input_features, output_features, true, "test_dense");
  layer->set_device(&getCPU());
  layer->initialize();

  // Create input
  Tensor<float> input({batch_size, input_features, 1, 1}, &getCPU());
  input.fill_random_uniform(1.0f);

  // Simulate 3 epochs with same data
  std::vector<Tensor<float>> outputs;
  for (int epoch = 0; epoch < 3; ++epoch) {
    const auto &output = layer->forward(input, 0);
    outputs.push_back(output.clone());
  }

  // All outputs should be identical (weights don't change without backward/optimize)
  for (size_t epoch = 1; epoch < outputs.size(); ++epoch) {
    ASSERT_EQ(outputs[0].shape(), outputs[epoch].shape());
    for (size_t i = 0; i < outputs[0].size(); ++i) {
      EXPECT_FLOAT_EQ(outputs[0].data_ptr().get()[i], outputs[epoch].data_ptr().get()[i])
          << "Mismatch at epoch " << epoch << " index " << i << ": "
          << outputs[0].data_ptr().get()[i] << " vs " << outputs[epoch].data_ptr().get()[i];
    }
  }
}

// Test Conv2D multiple passes to simulate epoch behavior
TEST(LayerBufferReuseTest, Conv2DMultipleEpochs) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 6;
  const size_t input_w = 6;

  auto layer = std::make_unique<Conv2DLayer<float>>(in_channels, out_channels, 3, 3, 1, 1, 0, 0,
                                                    true, "test_conv");
  layer->set_device(&getCPU());
  layer->initialize();

  // Create input
  Tensor<float> input({batch_size, in_channels, input_h, input_w}, &getCPU());
  input.fill_random_uniform(1.0f);

  // Simulate 3 epochs with same data
  std::vector<Tensor<float>> outputs;
  for (int epoch = 0; epoch < 3; ++epoch) {
    const auto &output = layer->forward(input, 0);
    outputs.push_back(output.clone());
  }

  // All outputs should be identical
  for (size_t epoch = 1; epoch < outputs.size(); ++epoch) {
    ASSERT_EQ(outputs[0].shape(), outputs[epoch].shape());
    for (size_t i = 0; i < outputs[0].size(); ++i) {
      EXPECT_FLOAT_EQ(outputs[0].data_ptr().get()[i], outputs[epoch].data_ptr().get()[i])
          << "Mismatch at epoch " << epoch << " index " << i << ": "
          << outputs[0].data_ptr().get()[i] << " vs " << outputs[epoch].data_ptr().get()[i];
    }
  }
}
