#include <iostream>

#include "data_loading/data_loader_factory.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/metrics.hpp"
#include "nn/optimizers.hpp"

using namespace std;
using namespace synet;

std::shared_ptr<Graph> make_mnist_model() {
  auto graph = make_shared<Graph>();

  auto input = graph->input("input");

  auto conv2d_1 = Conv2DLayer(1, 8, 5, 5, 1, 1, 0, 0, false, "conv1");
  auto bn1 = BatchNormLayer(8, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto pool1 = MaxPool2DLayer(3, 3, 3, 3, 0, 0, "pool1");
  auto conv2d_2 = Conv2DLayer(8, 16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1");
  auto bn2_1x1 = BatchNormLayer(16, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn2_1x1");
  auto conv2d_3 = Conv2DLayer(16, 48, 5, 5, 1, 1, 0, 0, false, "conv3");
  auto bn3 = BatchNormLayer(48, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn3");
  auto pool2 = MaxPool2DLayer(2, 2, 2, 2, 0, 0, "pool2");
  auto flatten = FlattenLayer(1, -1, "flatten");
  auto fc = DenseLayer(192, 10, false, "output");

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
  graph->set_output(output);

  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  graph->compile(allocator);

  return graph;
}

signed main() {
  cout << "Testing Graph API v2" << endl;

  auto graph = make_mnist_model();

  auto [train_loader, val_loader] = DataLoaderFactory::create("mnist", "data/mnist");
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for MNIST dataset" << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  Tensor data, labels;

  auto criterion = LossFactory::create_crossentropy(true, 1e-15);
  auto optimizer = OptimizerFactory::create_adam(0.001f, 0.9f, 0.999f);

  optimizer->attach(*graph);

  int epochs = 20;
  for (int i = 0; i < epochs; ++i) {
    train_loader->shuffle();
    train_loader->reset();

    graph->set_mode(ExecutionMode::TRAIN);
    int batch_idx = 0;
    // train
    while (train_loader->get_batch(256, data, labels)) {
      data = data.to_device(getGPU());
      labels = labels.to_device(getGPU());

      auto input_map = TensorBundle({
          {"input", data},
      });

      auto output_map = graph->forward(input_map);

      Tensor output = output_map.get("output");

      Tensor grad_output = Tensor(output.shape(), DType_t::FP32, output.device());

      float loss;
      criterion->compute_loss(output, labels, loss);
      criterion->compute_gradient(output, labels, grad_output);

      TensorBundle output_grad_map({
          {"output", grad_output},
      });

      auto input_grad_map = graph->backward(output_grad_map);

      optimizer->update();
      optimizer->zero_grads();

      cout << fmt::format("Batch: {}, Loss: {:.4f}", batch_idx, loss) << endl;

      ++batch_idx;
    }

    graph->set_mode(ExecutionMode::EVAL);
    val_loader->reset();
    // validation
    float total_val_loss = 0.0f;
    int val_samples = 0;
    int val_corrects = 0;
    while (val_loader->get_batch(256, data, labels)) {
      data = data.to_device(getGPU());
      labels = labels.to_device(getGPU());
      auto input_map = TensorBundle({
          {"input", data},
      });

      auto output_map = graph->forward(input_map);
      Tensor output = output_map.get("output");
      float val_loss;
      criterion->compute_loss(output, labels, val_loss);
      int corrects = compute_class_corrects(output, labels);
      val_corrects += corrects;
      total_val_loss += val_loss;
      val_samples += output.shape()[0];
    }
    float avg_val_loss = total_val_loss / val_samples;
    float val_accuracy = static_cast<float>(val_corrects) / val_samples;
    cout << fmt::format("Validation Loss: {:.4f}, Accuracy: {:.2f}%", avg_val_loss,
                        val_accuracy * 100)
         << endl;
  }

  return 0;
}