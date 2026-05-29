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

  auto conv2d_1 = make_layer<Conv2DLayerImpl>(3, 32, 3, 3, 1, 1, 1, 1, false, "test_conv2d");
  auto output1 = conv2d_1(input1);

  auto conv2d_2 = make_layer<Conv2DLayerImpl>(3, 32, 3, 3, 1, 1, 1, 1, false, "test_conv2d_2");
  auto output2 = conv2d_2(input2);

  auto output = output1 + output2;

  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  graph->compile(allocator);

  return graph;
}

std::shared_ptr<Graph> make_mnist_model() {
  auto graph = make_shared<Graph>();

  auto input = graph->make_node("input");

  auto conv2d_1 = make_layer<Conv2DLayerImpl>(1, 8, 5, 5, 1, 1, 0, 0, false, "conv1");
  auto bn1 = make_layer<BatchNormLayerImpl>(8, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto pool1 = make_layer<MaxPool2DLayerImpl>(3, 3, 3, 3, 0, 0, "pool1");
  auto conv2d_2 = make_layer<Conv2DLayerImpl>(8, 16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1");
  auto bn2_1x1 =
      make_layer<BatchNormLayerImpl>(16, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn2_1x1");
  auto conv2d_3 = make_layer<Conv2DLayerImpl>(16, 48, 5, 5, 1, 1, 0, 0, false, "conv3");
  auto bn3 = make_layer<BatchNormLayerImpl>(48, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn3");
  auto pool2 = make_layer<MaxPool2DLayerImpl>(2, 2, 2, 2, 0, 0, "pool2");
  auto flatten = make_layer<FlattenLayerImpl>(1, -1, "flatten");
  auto fc = make_layer<DenseLayerImpl>(192, 10, false, "output");

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

LayerRef<ResidualBlock> make_basic_residual_block(
    size_t in_channels, size_t out_channels, size_t stride = 1,
    const std::string &name = "basic_residual_block") {
  auto main_seq = std::make_unique<Sequential>(
      Conv2DLayerImpl(in_channels, out_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv1"),
      BatchNormLayerImpl(out_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn1"),
      Conv2DLayerImpl(out_channels, out_channels, 3, 3, 1, 1, 1, 1, false, name + "_conv2"),
      BatchNormLayerImpl(out_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn2"));

  std::unique_ptr<Sequential> shortcut_seq = nullptr;
  if (stride != 1 || in_channels != out_channels) {
    shortcut_seq =
        std::make_unique<Sequential>(Conv2DLayerImpl(in_channels, out_channels, 1, 1, stride, stride, 0,
                                                 0, false, name + "_shortcut_conv"),
                                     BatchNormLayerImpl(out_channels, dtype_eps(DType_t::FP32), 0.1f,
                                                    true, true, name + "_shortcut_bn"));
  }

  return make_layer<ResidualBlock>(std::move(main_seq), std::move(shortcut_seq), name);
}

LayerRef<ResidualBlock> make_bottleneck_residual_block(
    size_t in_channels, size_t mid_channels, size_t out_channels, size_t stride = 1,
    const std::string &name = "bottleneck_residual_block") {
  auto main_seq = std::make_unique<Sequential>(
      Conv2DLayerImpl(in_channels, mid_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv1"),
      BatchNormLayerImpl(mid_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn1"),
      Conv2DLayerImpl(mid_channels, mid_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv2"),
      BatchNormLayerImpl(mid_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn2"),
      Conv2DLayerImpl(mid_channels, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv3"),
      BatchNormLayerImpl(out_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn3"));

  std::unique_ptr<Sequential> shortcut_seq = nullptr;
  if (stride != 1 || in_channels != out_channels) {
    shortcut_seq =
        std::make_unique<Sequential>(Conv2DLayerImpl(in_channels, out_channels, 1, 1, stride, stride, 0,
                                                 0, false, name + "_shortcut_conv"),
                                     BatchNormLayerImpl(out_channels, dtype_eps(DType_t::FP32), 0.1f,
                                                    true, true, name + "_shortcut_bn"));
  }

  return make_layer<ResidualBlock>(std::move(main_seq), std::move(shortcut_seq), name);
}

std::shared_ptr<Graph> make_resnet50_model() {
  auto graph = make_shared<Graph>();

  auto input = graph->make_node("input");

  // Initial convolution layer
  auto conv1 = make_layer<Conv2DLayerImpl>(3, 64, 7, 7, 2, 2, 3, 3, true, "conv1");
  auto bn1 = make_layer<BatchNormLayerImpl>(64, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto maxpool = make_layer<MaxPool2DLayerImpl>(3, 3, 2, 2, 1, 1, "maxpool");

  // LayerImpl 1: 64 -> 256 (3 bottleneck blocks)
  auto layer1_block1 = make_bottleneck_residual_block(64, 64, 256, 1, "layer1_block1");
  auto layer1_block2 = make_bottleneck_residual_block(256, 64, 256, 1, "layer1_block2");
  auto layer1_block3 = make_bottleneck_residual_block(256, 64, 256, 1, "layer1_block3");

  // LayerImpl 2: 256 -> 512 (4 bottleneck blocks)
  auto layer2_block1 = make_bottleneck_residual_block(256, 128, 512, 2, "layer2_block1");
  auto layer2_block2 = make_bottleneck_residual_block(512, 128, 512, 1, "layer2_block2");
  auto layer2_block3 = make_bottleneck_residual_block(512, 128, 512, 1, "layer2_block3");
  auto layer2_block4 = make_bottleneck_residual_block(512, 128, 512, 1, "layer2_block4");

  // LayerImpl 3: 512 -> 1024 (6 bottleneck blocks)
  auto layer3_block1 = make_bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1");
  auto layer3_block2 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2");
  auto layer3_block3 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3");
  auto layer3_block4 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4");
  auto layer3_block5 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5");
  auto layer3_block6 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6");

  // LayerImpl 4: 1024 -> 2048 (3 bottleneck blocks)
  auto layer4_block1 = make_bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1");
  auto layer4_block2 = make_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2");
  auto layer4_block3 = make_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3");

  // Global average pooling and classification head
  auto avgpool = make_layer<AvgPool2DLayerImpl>(7, 7, 1, 1, 0, 0, "avgpool");
  auto flatten = make_layer<FlattenLayerImpl>(1, -1, "flatten");
  auto fc = make_layer<DenseLayerImpl>(2048, 1000, true, "fc");

  // Forward pass
  auto x = conv1(input);
  x = bn1(x);
  x = maxpool(x);

  // LayerImpl 1
  x = layer1_block1(x);
  x = layer1_block2(x);
  x = layer1_block3(x);

  // LayerImpl 2
  x = layer2_block1(x);
  x = layer2_block2(x);
  x = layer2_block3(x);
  x = layer2_block4(x);

  // LayerImpl 3
  x = layer3_block1(x);
  x = layer3_block2(x);
  x = layer3_block3(x);
  x = layer3_block4(x);
  x = layer3_block5(x);
  x = layer3_block6(x);

  // LayerImpl 4
  x = layer4_block1(x);
  x = layer4_block2(x);
  x = layer4_block3(x);

  // Classification head
  x = avgpool(x);
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

  auto [train_loader, val_loader] = DataLoaderFactory::create("mnist", "data/mnist");
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for MNIST dataset" << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  Tensor data, labels;

  auto criterion = LossFactory::create_crossentropy(true, 1e-15);
  auto optimizer = OptimizerFactory::create_adam(0.001f, 0.9f, 0.999f);

  optimizer->attach(*graph->context());

  int epochs = 20;
  for (int i = 0; i < epochs; ++i) {
    train_loader->shuffle();
    train_loader->reset();

    graph->set_mode(ExecutionMode::TRAIN);
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

    graph->set_mode(ExecutionMode::EVAL);
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