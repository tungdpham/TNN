#include <iostream>

#include "data_loading/data_loader_factory.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph_api.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/metrics.hpp"
#include "nn/node_ops.hpp"
#include "nn/optimizers.hpp"

using namespace std;
using namespace tnn;
using namespace tnn::graph_api_v2;

std::shared_ptr<Graph> make_model_test() {
  auto graph = make_shared<Graph>();
  auto input1 = graph->make_node("input1");
  auto input2 = graph->make_node("input2");

  auto conv2d_1 = make_layer<Conv2DLayer>(3, 32, 3, 3, 1, 1, 1, 1, false, "test_conv2d");
  auto output1 = conv2d_1(input1);

  auto conv2d_2 = make_layer<Conv2DLayer>(3, 32, 3, 3, 1, 1, 1, 1, false, "test_conv2d_2");
  auto output2 = conv2d_2(input2);

  auto output = output1 + output2;

  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  graph->compile(allocator);

  return graph;
}

std::shared_ptr<Graph> make_mnist_model() {
  auto graph = make_shared<Graph>();

  auto input = graph->make_node("input");

  auto conv2d_1 = make_layer<Conv2DLayer>(1, 8, 5, 5, 1, 1, 0, 0, false, "conv1");
  auto bn1 = make_layer<BatchNormLayer>(8, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto pool1 = make_layer<MaxPool2DLayer>(3, 3, 3, 3, 0, 0, "pool1");
  auto conv2d_2 = make_layer<Conv2DLayer>(8, 16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1");
  auto bn2_1x1 =
      make_layer<BatchNormLayer>(16, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn2_1x1");
  auto conv2d_3 = make_layer<Conv2DLayer>(16, 48, 5, 5, 1, 1, 0, 0, false, "conv3");
  auto bn3 = make_layer<BatchNormLayer>(48, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn3");
  auto pool2 = make_layer<MaxPool2DLayer>(2, 2, 2, 2, 0, 0, "pool2");
  auto flatten = make_layer<FlattenLayer>(1, -1, "flatten");
  auto fc = make_layer<DenseLayer>(192, 10, false, "output");

  auto x = conv2d_1(input);
  x = bn1(x);
  x = pool1(x);
  x = conv2d_2(x);
  x = bn2_1x1(x);
  x = conv2d_3(x);
  x = bn3(x);
  x = pool2(x);
  x = flatten(x);
  auto output = fc(x);
  output->set_uid("output");

  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  graph->compile(allocator);

  return graph;
}

signed main() {
  cout << "Testing Graph API v2" << endl;

  auto graph = make_mnist_model();

  auto batch_data = make_tensor<float>({1, 64, 64, 3}, getGPU());
  batch_data->fill(1.0f);  // Fill with dummy data

  auto [train_loader, val_loader] = DataLoaderFactory::create("mnist", "data/mnist");
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for MNIST dataset" << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  Tensor data, labels;

  auto criterion = LossFactory::create_crossentropy(true, 1e-15);
  auto optimizer = OptimizerFactory::create_adam(0.001f, 0.9f, 0.999f, 1e-8f, 1e-4f);

  optimizer->attach(*graph->context());

  int epochs = 20;
  for (int i = 0; i < epochs; ++i) {
    train_loader->shuffle();
    train_loader->reset();

    int batch_idx = 0;
    // train
    while (train_loader->get_batch(256, data, labels)) {
      data = data->to_device(getGPU());
      labels = labels->to_device(getGPU());

      auto input_map = InputMap({
          {"input", data},
      });

      auto output_map = graph->forward(input_map);

      Tensor output = output_map.get("output");

      Tensor grad_output = make_tensor<float>(output->shape(), output->device());

      float loss;
      criterion->compute_loss(output, labels, loss);
      criterion->compute_gradient(output, labels, grad_output);

      InputMap output_grad_map({
          {"output", grad_output},
      });

      auto input_grad_map = graph->backward(output_grad_map);

      optimizer->update();

      cout << fmt::format("Batch: {}, Loss: {:.4f}", batch_idx, loss) << endl;

      ++batch_idx;
    }

    val_loader->reset();
    // validation
    float total_val_loss = 0.0f;
    int val_samples = 0;
    int val_corrects = 0;
    while (val_loader->get_batch(256, data, labels)) {
      data = data->to_device(getGPU());
      labels = labels->to_device(getGPU());
      auto input_map = InputMap({
          {"input", data},
      });

      auto output_map = graph->forward(input_map);
      Tensor output = output_map.get("output");
      float val_loss;
      criterion->compute_loss(output, labels, val_loss);
      int corrects = compute_class_corrects(output, labels);
      val_corrects += corrects;
      total_val_loss += val_loss;
      val_samples += output->shape()[0];
    }
    float avg_val_loss = total_val_loss / val_samples;
    float val_accuracy = static_cast<float>(val_corrects) / val_samples;
    cout << fmt::format("Validation Loss: {:.4f}, Accuracy: {:.2f}%", avg_val_loss,
                        val_accuracy * 100)
         << endl;
  }

  return 0;
}