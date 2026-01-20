/*
 * Test to verify that layers properly handle buffer reuse
 * across multiple forward/backward passes
 */

#include <gtest/gtest.h>
#include <memory>

#include "nn/activations.hpp"
#include "nn/layers_impl/activation_layer.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/flatten_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "nn/layers_impl/legacy_maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

TEST(LayerBufferReuseTest, Conv2DConsistentOutput) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 4;
  const size_t input_h = 8;
  const size_t input_w = 8;

  auto layer = std::make_unique<LegacyConv2DLayer>(in_channels, out_channels, 3, 3, 1, 1, 1, 1,
                                                   true, "test_conv");
  layer->set_device(getCPU());
  layer->init();

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  Tensor output1 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output1, 0);
  Tensor output1_copy = output1->clone();

  Tensor output2 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output2, 0);

  ASSERT_EQ(output1_copy->shape(), output2->shape());
  for (size_t i = 0; i < output1_copy->size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy->data_as<float>()[i], output2->data_as<float>()[i])
        << "Mismatch at index " << i;
  }
}

TEST(LayerBufferReuseTest, DenseConsistentOutput) {
  const size_t batch_size = 4;
  const size_t input_features = 32;
  const size_t output_features = 10;

  auto layer = std::make_unique<DenseLayer>(input_features, output_features, true, "test_dense");
  layer->set_device(getCPU());
  layer->init();

  Tensor input = make_tensor<float>({batch_size, input_features, 1, 1}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  Tensor output1 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output1, 0);
  Tensor output1_copy = output1->clone();

  Tensor output2 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output2, 0);

  ASSERT_EQ(output1_copy->shape(), output2->shape());
  for (size_t i = 0; i < output1_copy->size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy->data_as<float>()[i], output2->data_as<float>()[i])
        << "Mismatch at index " << i << ": " << output1_copy->data_as<float>()[i] << " vs "
        << output2->data_as<float>()[i];
  }
}

TEST(LayerBufferReuseTest, MaxPool2DConsistentOutput) {
  const size_t batch_size = 2;
  const size_t channels = 8;
  const size_t input_h = 8;
  const size_t input_w = 8;

  auto layer = std::make_unique<LegacyMaxPool2DLayer>(2, 2, 2, 2, 0, 0, "test_pool");
  layer->set_device(getCPU());

  Tensor input = make_tensor<float>({batch_size, channels, input_h, input_w}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  Tensor output1 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output1, 0);
  Tensor output1_copy = output1->clone();

  Tensor output2 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output2, 0);

  ASSERT_EQ(output1_copy->shape(), output2->shape());
  for (size_t i = 0; i < output1_copy->size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy->data_as<float>()[i], output2->data_as<float>()[i])
        << "Mismatch at index " << i;
  }
}

TEST(LayerBufferReuseTest, ActivationConsistentOutput) {
  const size_t batch_size = 2;
  const size_t channels = 8;
  const size_t h = 4;
  const size_t w = 4;

  auto factory = ActivationFactory();
  factory.register_defaults();
  auto activation = factory.create("relu");
  auto layer = std::make_unique<ActivationLayer>(std::move(activation), "test_relu");
  layer->set_device(getCPU());

  Tensor input = make_tensor<float>({batch_size, channels, h, w}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  Tensor output1 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output1, 0);
  Tensor output1_copy = output1->clone();

  Tensor output2 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output2, 0);

  ASSERT_EQ(output1_copy->shape(), output2->shape());
  for (size_t i = 0; i < output1_copy->size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy->data_as<float>()[i], output2->data_as<float>()[i])
        << "Mismatch at index " << i;
  }
}

TEST(LayerBufferReuseTest, FlattenConsistentOutput) {
  const size_t batch_size = 2;
  const size_t channels = 8;
  const size_t h = 4;
  const size_t w = 4;

  auto layer = std::make_unique<FlattenLayer>(1, "test_flatten");
  layer->set_device(getCPU());

  Tensor input = make_tensor<float>({batch_size, channels, h, w}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  Tensor output1 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output1, 0);
  Tensor output1_copy = output1->clone();

  Tensor output2 = make_tensor<float>(output_shape, &getCPU());
  layer->forward(input, output2, 0);

  ASSERT_EQ(output1_copy->shape(), output2->shape());
  for (size_t i = 0; i < output1_copy->size(); ++i) {
    EXPECT_FLOAT_EQ(output1_copy->data_as<float>()[i], output2->data_as<float>()[i])
        << "Mismatch at index " << i;
  }
}

TEST(LayerBufferReuseTest, DenseMultipleEpochs) {
  const size_t batch_size = 4;
  const size_t input_features = 16;
  const size_t output_features = 8;

  auto layer = std::make_unique<DenseLayer>(input_features, output_features, true, "test_dense");
  layer->set_device(getCPU());
  layer->init();

  Tensor input = make_tensor<float>({batch_size, input_features, 1, 1}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  std::vector<Tensor> outputs;
  for (int epoch = 0; epoch < 3; ++epoch) {
    Tensor output = make_tensor<float>(output_shape, &getCPU());
    layer->forward(input, output, 0);
    outputs.push_back(output->clone());
  }

  for (size_t epoch = 1; epoch < outputs.size(); ++epoch) {
    ASSERT_EQ(outputs[0]->shape(), outputs[epoch]->shape());
    for (size_t i = 0; i < outputs[0]->size(); ++i) {
      EXPECT_FLOAT_EQ(outputs[0]->data_as<float>()[i], outputs[epoch]->data_as<float>()[i])
          << "Mismatch at epoch " << epoch << " index " << i << ": "
          << outputs[0]->data_as<float>()[i] << " vs " << outputs[epoch]->data_as<float>()[i];
    }
  }
}

TEST(LayerBufferReuseTest, Conv2DMultipleEpochs) {
  const size_t batch_size = 2;
  const size_t in_channels = 3;
  const size_t out_channels = 8;
  const size_t input_h = 6;
  const size_t input_w = 6;

  auto layer = std::make_unique<LegacyConv2DLayer>(in_channels, out_channels, 3, 3, 1, 1, 0, 0,
                                                   true, "test_conv");
  layer->set_device(getCPU());
  layer->init();

  Tensor input = make_tensor<float>({batch_size, in_channels, input_h, input_w}, &getCPU());
  input->fill_random_uniform(1.0f);

  std::vector<size_t> output_shape = layer->compute_output_shape(input->shape());
  std::vector<Tensor> outputs;
  for (int epoch = 0; epoch < 3; ++epoch) {
    Tensor output = make_tensor<float>(output_shape, &getCPU());
    layer->forward(input, output, 0);
    outputs.push_back(output->clone());
  }

  for (size_t epoch = 1; epoch < outputs.size(); ++epoch) {
    ASSERT_EQ(outputs[0]->shape(), outputs[epoch]->shape());
    for (size_t i = 0; i < outputs[0]->size(); ++i) {
      EXPECT_FLOAT_EQ(outputs[0]->data_as<float>()[i], outputs[epoch]->data_as<float>()[i])
          << "Mismatch at epoch " << epoch << " index " << i << ": "
          << outputs[0]->data_as<float>()[i] << " vs " << outputs[epoch]->data_as<float>()[i];
    }
  }
}
